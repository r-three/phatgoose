import re
import string

import datasets
import evaluate
from evaluate import load


class ExactMatchMultipleAns(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description="Exact match with multiple answer choices available",
            citation="",
            inputs_description="",
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string", id="sequence"),
                    "references": datasets.Sequence(
                        datasets.Value("string", id="sequence")
                    ),
                }
            ),
        )

    def _compute(self, predictions, references):
        num_matches = 0
        for pred, ref_list in zip(predictions, references):
            if pred in ref_list:
                num_matches += 1

        exact_match_score = num_matches / len(predictions)
        return {"exact_match_multiple_ans": exact_match_score}


class AccuracyMulitpleAns(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description="Accuracy with multiple answer choices available",
            citation="",
            inputs_description="",
            features=datasets.Features(
                {
                    "predictions": datasets.Value("int32"),
                    "references": datasets.Sequence(datasets.Value("int32")),
                }
            ),
        )

    def _compute(self, predictions, references):
        num_matches = 0
        for pred, ref_list in zip(predictions, references):
            if pred in ref_list:
                num_matches += 1

        accuracy_score = num_matches / len(predictions)
        return {"accuracy_multiple_ans": accuracy_score}


def convert_dict_of_lists_to_list_of_dicts(dict_of_lists):
    list_of_dicts = []
    for datapoint_values in zip(*dict_of_lists.values()):
        list_of_dicts.append(dict(zip(dict_of_lists, datapoint_values)))
    return list_of_dicts


class Scorer(object):
    def __init__(self, metrics):
        self.metrics_to_compute = {
            "accuracy": False,
            "squad": False,
            "matthews_correlation": False,
            "f1": False,
            "pearsonr": False,
            "spearmanr": False,
            "bleu": False,
            "bertscore": False,
            "rouge": False,
            "exact_match": False,
            "custom": False,
            "exact_match_multiple_ans": False,
            "accuracy_multiple_ans": False,
        }

        if "accuracy" in metrics:
            self.metrics_to_compute["accuracy"] = True
            self.accuracy_metric = load("accuracy", keep_in_memory=True)

        if "squad" in metrics:
            self.metrics_to_compute["squad"] = True
            self.squad_metric = load("squad", keep_in_memory=True)

        if "matthews_correlation" in metrics:
            self.metrics_to_compute["matthews_correlation"] = True
            self.matthews_correlation_metric = load(
                "matthews_correlation", keep_in_memory=True
            )

        if "f1" in metrics:
            self.metrics_to_compute["f1"] = True
            self.f1_metric = load("f1", keep_in_memory=True)

        if "pearsonr" in metrics:
            self.metrics_to_compute["pearsonr"] = True
            self.pearsons_correlation_metric = load("pearsonr", keep_in_memory=True)

        if "spearmanr" in metrics:
            self.metrics_to_compute["spearmanr"] = True
            self.spearmanr_correlation_metric = load("spearmanr", keep_in_memory=True)

        if "bleu" in metrics:
            self.metrics_to_compute["bleu"] = True
            self.bleu_metric = load("bleu", keep_in_memory=True)

        if "bertscore" in metrics:
            self.metrics_to_compute["bertscore"] = True
            self.bertscore_metric = load("bertscore", keep_in_memory=True)

        if "rouge" in metrics:
            self.metrics_to_compute["rouge"] = True
            self.rouge_metric = load("rouge", keep_in_memory=True)

        if "exact_match" in metrics:
            self.metrics_to_compute["exact_match"] = True
            self.exact_match_metric = load("exact_match", keep_in_memory=True)

        if "custom" in metrics:
            self.metrics_to_compute["custom"] = True
            self.custom_metric = load("exact_match", keep_in_memory=True)

        if "exact_match_multiple_ans" in metrics:
            self.metrics_to_compute["exact_match_multiple_ans"] = True
            self.exact_match_multiple_ans_metric = ExactMatchMultipleAns()

        if "accuracy_multiple_ans" in metrics:
            self.metrics_to_compute["accuracy_multiple_ans"] = True
            self.accuracy_multiple_ans_metric = AccuracyMulitpleAns()

    def add_batch(self, batch_inputs, batch_outputs):
        if self.metrics_to_compute["accuracy"]:
            self.accuracy_metric.add_batch(
                predictions=batch_outputs["prediction"],
                references=batch_inputs["label"],
            )

        if self.metrics_to_compute["matthews_correlation"]:
            self.matthews_correlation_metric.add_batch(
                predictions=batch_outputs["prediction"],
                references=batch_inputs["label"],
            )

        if self.metrics_to_compute["f1"]:
            self.f1_metric.add_batch(
                predictions=batch_outputs["prediction"],
                references=batch_inputs["label"],
            )

        if self.metrics_to_compute["pearsonr"]:
            self.pearsons_correlation_metric.add_batch(
                predictions=batch_outputs["prediction"],
                references=batch_inputs["label"],
            )

        if self.metrics_to_compute["spearmanr"]:
            self.spearmanr_correlation_metric.add_batch(
                predictions=batch_outputs["prediction"],
                references=batch_inputs["label"],
            )

        if self.metrics_to_compute["bleu"]:
            self.bleu_metric.add_batch(
                predictions=batch_outputs["output_text"],
                references=batch_inputs["references"],
            )

        if self.metrics_to_compute["bertscore"]:
            self.bertscore_metric.add_batch(
                predictions=batch_outputs["output_text"],
                references=batch_inputs["references"],
            )

        if self.metrics_to_compute["rouge"]:
            self.rouge_metric.add_batch(
                predictions=batch_outputs["output_text"],
                references=batch_inputs["references"],
            )

        if self.metrics_to_compute["squad"]:
            # TODO: verify if its right
            self.squad_metric.add_batch(
                predictions=convert_dict_of_lists_to_list_of_dicts(
                    {
                        "id": [str(idx) for idx in batch_inputs["example_idx"]],
                        "prediction_text": batch_outputs["output_text"],
                    }
                ),
                references=convert_dict_of_lists_to_list_of_dicts(
                    {
                        "id": [str(idx) for idx in batch_inputs["example_idx"]],
                        "answers": [
                            {"text": text, "answer_start": start}
                            for text, start in zip(
                                batch_inputs["references"],
                                batch_inputs["_answer_start"],
                            )
                        ],
                    }
                ),
            )

        if self.metrics_to_compute["exact_match"]:
            # predictions and references are list of strings
            self.exact_match_metric.add_batch(
                predictions=batch_outputs["output_text"],
                references=batch_inputs["references"],
            )

        if self.metrics_to_compute["custom"]:
            self.custom_metric.add_batch(
                predictions=batch_outputs["output_text"],
                references=batch_outputs["output_text"],
            )

        if self.metrics_to_compute["exact_match_multiple_ans"]:
            self.exact_match_multiple_ans_metric.add_batch(
                predictions=batch_outputs["output_text"],
                references=batch_inputs["references"],
            )

        if self.metrics_to_compute["accuracy_multiple_ans"]:
            self.accuracy_multiple_ans_metric.add_batch(
                predictions=batch_outputs["prediction"],
                references=batch_inputs["multi_label"],
            )

    def get_score(self):
        score = {}

        if self.metrics_to_compute["accuracy"]:
            score.update(self.accuracy_metric.compute())

        if self.metrics_to_compute["squad"]:
            squad_metrics = self.squad_metric.compute()
            # Scale SQUAD metrics to be between 0 and 1
            for metric, value in squad_metrics.items():
                squad_metrics[metric] = value / 100
            score.update(squad_metrics)

        if self.metrics_to_compute["matthews_correlation"]:
            score.update(self.matthews_correlation_metric.compute())

        if self.metrics_to_compute["f1"]:
            score.update(self.f1_metric.compute())

        if self.metrics_to_compute["pearsonr"]:
            score.update(self.pearsons_correlation_metric.compute())

        if self.metrics_to_compute["spearmanr"]:
            score.update(self.spearmanr_correlation_metric.compute())

        if self.metrics_to_compute["bleu"]:
            bleu_metrics = self.bleu_metric.compute()
            score.update(
                {key: bleu_metrics[key] for key in bleu_metrics if key in ["bleu"]}
            )

        if self.metrics_to_compute["bertscore"]:
            bertscore_metrics = self.bertscore_metric.compute(
                model_type="microsoft/deberta-large-mnli"
            )
            score.update(
                {
                    "bertscore": sum(bertscore_metrics["precision"])
                    / len(bertscore_metrics["precision"])
                }
            )

        if self.metrics_to_compute["rouge"]:
            score.update(self.rouge_metric.compute(rouge_types=["rougeL"]))

        if self.metrics_to_compute["exact_match"]:
            score.update(self.exact_match_metric.compute())

        if self.metrics_to_compute["custom"]:
            score.update(self.custom_metric.compute())

        if self.metrics_to_compute["exact_match_multiple_ans"]:
            score.update(self.exact_match_multiple_ans_metric.compute())

        if self.metrics_to_compute["accuracy_multiple_ans"]:
            score.update(self.accuracy_multiple_ans_metric.compute())

        for key, value in score.items():
            score[key] = float("%.3f" % value)

        return score
