import os
import re
from collections import defaultdict

import gin
import numpy as np
import torch

import src.utils.logging as logging
from src.procedures.procedure import Procedure
from src.procedures.utils.batcher import MultiTaskBatcher, SingleTaskBatcher
from src.procedures.utils.optimizer_scheduler import get_optimizer, get_scheduler


# Make the order consistent and reasonable
@gin.configurable(
    allowlist=[
        "model",
        "datasets",
        "batcher",
        "training_tracker",
        "num_steps",
        "gradient_accumulation_factor",
        "gradient_clipping",
        "validate_procedure",
        "report_step_interval",
        "validation_step_interval",
        "save_model_step_interval",
        "checkpoint_step_interval",
        "step_moma_calls",
        "report_moma_calls",
        "save_model_moma_calls",
        "finish_moma_calls",
        "pass_current_step",
        "stiefel",
    ],
)
class Trainer(Procedure):
    linking_fields = ["model", "datasets", "validate_procedure"]

    def __init__(
        self,
        model,
        datasets,
        batcher=SingleTaskBatcher(shuffle=True, drop_last=True, num_workers=8),
        num_steps=10000,
        gradient_accumulation_factor=1,
        gradient_clipping=1.0,
        validate_procedure=None,
        report_step_interval=50,
        validation_step_interval=None,
        save_model_step_interval=None,
        checkpoint_step_interval=None,
        step_moma_calls=[],
        report_moma_calls=[],
        save_model_moma_calls=[],
        finish_moma_calls=[],
        pass_current_step=False,
        stiefel=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.datasets = datasets
        self.validate_procedure = validate_procedure

        self.num_steps = num_steps
        self.gradient_accumulation_factor = gradient_accumulation_factor
        self.gradient_clipping = gradient_clipping

        self.batcher = batcher
        self.loss_scaler = None
        self.optimizer = None
        self.scheduler = None
        self.report_tracker = Tracker()

        self.step_moma_calls = step_moma_calls
        self.report_moma_calls = report_moma_calls
        self.save_model_moma_calls = save_model_moma_calls
        self.finish_moma_calls = finish_moma_calls

        self.report_step_interval = report_step_interval
        self.validation_step_interval = validation_step_interval
        self.save_model_step_interval = save_model_step_interval
        self.checkpoint_step_interval = checkpoint_step_interval

        self.current_step = 0
        self.pass_current_step = pass_current_step
        self.stiefel = stiefel

        if self.save_model_step_interval is not None and save_model_moma_calls == []:
            raise NotImplementedError()
            # TODO: provide a default save_model function
        if self.checkpoint_step_interval is not None:
            assert (
                self.save_model_step_interval is not None
                and self.checkpoint_step_interval % self.save_model_step_interval == 0
            ), "checkpoint_step_interval need to be a multiple of save_model_step_interval"

    def link(self):
        if isinstance(self.datasets, str):
            self.datasets = [self.datasets]
        super().link()

    def late_init(self):
        self.optimizer = get_optimizer(
            self.model,
        )
        self.scheduler = get_scheduler(
            optimizer=self.optimizer,
            num_steps=self.num_steps,
        )
        if self.model.mix_precision in ["bf16", "fp16"]:
            self.loss_scaler = torch.cuda.amp.GradScaler()
        for dataset in self.datasets:
            dataset.set_tokenizer(self.model.tokenizer)

        self.batcher.set_tokenizer(self.model.tokenizer)
        self.batcher.set_seed(self.seed)
        self._data_loader = self.batcher.build(self.datasets)

    def _get_train_batches(self):
        while True:
            for batch_inputs in self._data_loader:
                yield batch_inputs

    def prepare_passing_global_hiddens(self):
        passing_global_hiddens = {}
        if self.pass_current_step:
            passing_global_hiddens["current_step"] = self.current_step
        return passing_global_hiddens

    def _update_router(self, update_gradient=True):
        # update gradient to make it orthogonal to frozen expert embeddings
        def gram_schmidt(vectors):
            basis = []
            for v in vectors:
                w = v - sum(torch.dot(v, u) / torch.dot(u, u) * u for u in basis)
                if torch.norm(w) > 1e-6:  # Avoid adding zero-length vectors
                    basis.append(w)
            return basis

        regex_pattern = r"(.*)__(\d+)$"
        grouped_params = {}

        for key, param in self.model.torch_model.named_parameters():
            if "expert_embeddings" in key:
                match = re.match(regex_pattern, key)
                if match:
                    prefix = match.group(1)
                    index = match.group(2)
                    if prefix not in grouped_params:
                        grouped_params[prefix] = {}
                    grouped_params[prefix][index] = param
        update = {}
        for prefix in grouped_params:
            other_vectors = []
            original_vector = None
            original_vector_key_name = None
            for index in grouped_params[prefix]:
                key_name = f"{prefix}__{index}"
                param = grouped_params[prefix][index]
                if param.grad is None:
                    other_vectors.append(param.data.float())
                else:
                    if update_gradient:
                        original_vector = param.grad
                    else:
                        original_vector = param.data.float()
                    original_vector_key_name = key_name
            basis = gram_schmidt(other_vectors)
            orthogonal_subspace_vector = original_vector - sum(
                torch.dot(original_vector, u) / torch.dot(u, u) * u for u in basis
            )
            update[original_vector_key_name] = orthogonal_subspace_vector
        for key, param in self.model.torch_model.named_parameters():
            if key in update:
                # print(f"updating {key}")
                if update_gradient:
                    param.grad = update[key]
                else:
                    param.data = update[key]
        # group expert_embeddings to form a matrix and call it W
        # group grad of expert_embeddings to form a matrix and call it G
        # # check for infs and NaNs and if so don't update
        # for name, param in named_trainable_parameters:
        #     if "expert_embeddings" in name and param.grad is not None:
        # A = G@W - W@G
        # U = A@W
        # def matrix_norm_one(W):
        #     out = torch.abs(W)
        #     out = torch.sum(out, dim=0)
        #     out = torch.max(out)
        #     return out
        # tau = min(lr, 1/matrix_norm_one(W))
        # Y = W - tau*U
        # for i in range(1,3):
        #     Y = W - tau * 0.5 * A @ (W + Y)
        # # uncouple Y to expert_embeddings
        # return Y

    def run(self):
        best_result = None
        logging.print_single_bar()
        print(f"Running {self.name}...")
        self.model.torch_model.train()
        data_iter = self._get_train_batches()
        while self.current_step < self.num_steps:
            self.optimizer.zero_grad()
            for _ in range(self.gradient_accumulation_factor):
                batch_inputs = next(data_iter)
                # TODO: Assuming all datasets have the same interface during training i.e lm
                batch_dataset = self.datasets[0]
                batch_outputs = self.model(
                    batch_inputs,
                    batch_dataset.interface_info,
                    self.prepare_passing_global_hiddens(),
                )
                loss = batch_outputs["loss"]
                scaled_loss = loss / self.gradient_accumulation_factor
                if self.loss_scaler is not None:
                    scaled_loss = self.loss_scaler.scale(scaled_loss)
                scaled_loss.backward()

                self.report_tracker.add(
                    loss=loss,
                    global_hidden_dict=self.model.global_hidden_dict,
                )

            if self.loss_scaler is not None:
                self.loss_scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.named_trainable_parameters().values(),
                self.gradient_clipping,
                error_if_nonfinite=False,
            )
            self.report_tracker.add(
                grad_norm=grad_norm,
                lr=self.optimizer.param_groups[0]["lr"],
            )
            for moma_call in self.step_moma_calls:
                moma_call(self.model)
            if self.stiefel:
                self._update_router(update_gradient=True)
            if self.loss_scaler is not None:
                self.loss_scaler.step(self.optimizer)
                self.loss_scaler.update()
            else:
                self.optimizer.step()
            if self.stiefel:
                self._update_router(update_gradient=False)
            if self.scheduler is not None:
                self.scheduler.step()
            self.current_step += 1
            logging.logger_step()
            if self.current_step % self.report_step_interval == 0:
                for moma_call in self.report_moma_calls:
                    moma_call(self.model)
                report = self.report_tracker.get_summary()
                logging.log_scalar_dict(report)
                print(f"\tStep {logging.global_step}: {report}")

            if (
                self.validation_step_interval
                and self.current_step % self.validation_step_interval == 0
            ):
                self.validate_procedure.run(logging.global_step)
                logging.print_single_bar()
                print(f"Return to {self.name}...")
                self.model.torch_model.train()
            if (
                self.save_model_step_interval
                and self.current_step % self.save_model_step_interval == 0
            ):
                for moma_call in self.save_model_moma_calls:
                    moma_call(self.model)
            if (
                self.checkpoint_step_interval
                and self.current_step % self.checkpoint_step_interval == 0
            ):
                self.save_states()

        print(f"Best result: {best_result}")
        print(f"Finished {self.name}")
        for moma_call in self.finish_moma_calls:
            moma_call(self.model)

    def save_states(self, checkpoint_path):
        pass

    def recover_states(self, checkpoint_path):
        pass

    def get_description(self):
        return [
            f"Procedure class: {self.__class__.__name__}",
            f"Train {self.model.name} model on {len(self.datasets)} datasets ({[dataset.name for dataset in self.datasets]})",
            f"Optimizer: {self.optimizer}",
            f"Scheduler: {self.scheduler}",
            f"{self.num_steps} step x {self.gradient_accumulation_factor} grad accumulations",
        ]


@gin.configurable(allowlist=["global_hidden_to_keep"])
class Tracker:
    def __init__(self, global_hidden_to_keep=["loss", "scale"]):
        self.global_hidden_to_keep = global_hidden_to_keep
        self._results = defaultdict(list)

    def add(self, loss=None, global_hidden_dict=None, grad_norm=None, lr=None):
        if loss is not None:
            self._results["loss"].append(loss.detach().cpu().item())
        if grad_norm is not None:
            self._results["grad_norm"].append(grad_norm.detach().cpu().item())
        if lr is not None:
            self._results["lr"].append(lr)
        if global_hidden_dict is not None:
            for key, value in global_hidden_dict.items():
                if key[0] in self.global_hidden_to_keep:
                    report_key = f"{key[0]}/{'.'.join(key[1:])}"
                    self._results[report_key].append(value.detach().cpu().item())

    def get_summary(self, clear=True):
        summary = {}
        for key, value in self._results.items():
            value = np.mean(value).item()
            if value > 1e-4:
                summary[key] = round(value, 4)
            else:
                summary[key] = value
        if clear:
            self._results.clear()
        return summary
