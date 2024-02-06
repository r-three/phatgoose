import json
import os
from collections import OrderedDict

import gin
import torch

import src.utils.logging as logging
from src.data.metrics import Scorer
from src.procedures.procedure import Procedure
from src.procedures.utils.batcher import SingleTaskBatcher
from src.procedures.utils.result_aggregators import MainAggregator


@gin.configurable(
    allowlist=[
        "model",
        "datasets",
        "save_results",
        "batcher",
        "results_aggregators",
        "analysis_processors",
        "higher_is_better",
        "better_model_moma_calls",
    ]
)
# Set the random seed
class Evaluator(Procedure):
    linking_fields = ["model", "datasets"]

    def __init__(
        self,
        model,
        datasets,
        save_results,
        batcher=SingleTaskBatcher(shuffle=False, drop_last=False, num_workers=8),
        results_aggregators=[MainAggregator()],
        analysis_processors=[],
        higher_is_better=True,
        better_model_moma_calls=[],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.datasets = datasets
        self.batcher = batcher
        self.save_results = save_results
        self.results_aggregators = results_aggregators
        self._current_results = OrderedDict()
        self.scorer = OrderedDict()
        self.analysis_processors = analysis_processors
        self.higher_is_better = higher_is_better
        self.better_model_moma_calls = better_model_moma_calls
        self.best_results = None

    def link(self):
        if isinstance(self.datasets, str):
            self.datasets = [self.datasets]
        super().link()

    def late_init(self):
        for dataset in self.datasets:
            dataset.set_tokenizer(self.model.tokenizer)
            if dataset.metrics is not None:
                self.scorer[dataset.name] = Scorer(dataset.metrics)
        self.batcher.set_tokenizer(self.model.tokenizer)
        self.batcher.set_seed(self.seed)

    def run(self, step=None):
        logging.print_single_bar()
        print(f"Running {self.name}...")
        self.model.torch_model.eval()
        with torch.no_grad():
            for dataset in self.datasets:
                print(f"\tEvaluating {dataset.name}...")
                if dataset.name in self._current_results:
                    continue
                if dataset.metrics is None:
                    continue
                data_loader = self.batcher.build(dataset)
                for batch_idx, batch_inputs in enumerate(data_loader):
                    # data_dict = {}
                    # input_tokens = self.model.tokenizer.tokenize(batch_inputs['input_str'][0])
                    # print(f"first {len(input_tokens)}: {input_tokens}\n")
                    # data_dict['example_first'] = input_tokens
                    # input_tokens = self.model.tokenizer.tokenize(batch_inputs['input_str'][-1])
                    # print(f"last {len(input_tokens)}: {input_tokens}\n")
                    # data_dict['example_last'] = input_tokens
                    # batch_outputs = self.model(batch_inputs, dataset.interface_info, data_dict)
                    batch_outputs = self.model(batch_inputs, dataset.interface_info, {})
                    self.scorer[dataset.name].add_batch(batch_inputs, batch_outputs)
                    for analysis_processor in self.analysis_processors:
                        analysis_processor.batch_process(
                            batch_inputs, batch_outputs, self.model.global_hidden_dict
                        )

                self._current_results[dataset.name] = self.scorer[
                    dataset.name
                ].get_score()
                self._current_results[dataset.name]["score"] = sum(
                    self._current_results[dataset.name].values()
                ) / len(self._current_results[dataset.name])
                self.save_states()

                for analysis_processor in self.analysis_processors:
                    analysis_processor.dataset_process(dataset.name)
                print(
                    f"\t{dataset.name} results: {self._current_results[dataset.name]}"
                )

        for aggregator in self.results_aggregators:
            aggregator(self._current_results)
        results = self._current_results.copy()
        if step is not None:
            results["step"] = step
        self._current_results.clear()

        print(f"\tAll results: {results}")
        print(f"Finished {self.name}")

        logging.log_scalar_dict(
            {f"{self.name}/{key}": value for key, value in results.items()}
        )
        self.save_results(results, step=step)
        for analysis_processor in self.analysis_processors:
            analysis_processor.cross_dataset_process()
            analysis_processor.save(step)

        if (
            self.best_results is None
            or (
                results["average_score"] >= self.best_results["average_score"]
                and self.higher_is_better
            )
            or (
                results["average_score"] <= self.best_results["average_score"]
                and not self.higher_is_better
            )
        ):
            print("\t New best results!")
            self.best_results = results
            for moma_call in self.better_model_moma_calls:
                moma_call(self.model)
        return results

    def save_states(self):
        # TODO(Checkpointing): save results and rng state
        pass

    def recover_states(self):
        # TODO(Checkpointing): load results and rng state
        pass

    def get_description(self):
        return [
            f"Procedure class: {self.__class__.__name__}",
            f"Evalutes {self.model.name} model on {len(self.datasets)} datasets ({[dataset.name for dataset in self.datasets]})",
        ]
