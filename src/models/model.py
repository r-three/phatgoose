from collections import OrderedDict

import gin
import torch

from src.models.addon_control_mixin import AddonControlMixin
from src.models.interface_mixin import InterfaceMixin
from src.models.manipulations.basic import (
    set_device_and_parallelism,
    set_trainable_params,
)


@gin.configurable(
    allowlist=[
        "torch_model",
        "tokenizer",
        "init_moma_calls",
        "trainable_params",
        "mix_precision",
        "parallelism",
        "device",
        "pass_batch_input",
    ]
)
class Model(AddonControlMixin, InterfaceMixin):
    def __init__(
        self,
        name,
        torch_model,
        tokenizer,
        trainable_params="all",
        mix_precision=None,
        parallelism=None,
        device="auto",
        init_moma_calls=[],
        pass_batch_input=False,
        *args,
        **kwargs,
    ):
        self.global_hidden_dict = OrderedDict()
        self.name = name
        self.torch_model = torch_model
        self.tokenizer = tokenizer
        self.trainable_params = trainable_params
        self.mix_precision = mix_precision
        self.parallelism = parallelism
        self.device = device
        self.pass_batch_input = pass_batch_input
        super().__init__(*args, **kwargs)

        for moma_call in init_moma_calls:
            moma_call(self)
        set_trainable_params(
            self,
            trainable_params=self.trainable_params,
            mix_precision=self.mix_precision,
        )
        set_device_and_parallelism(
            self,
            device=self.device,
            parallelism=self.parallelism,
        )

    def _clear_hiddens(self):
        """
        Clear the global_hidden_states.
        This should be called at the beginning of each forward pass.
        """
        self.global_hidden_dict.clear()

    def __call__(self, batch_input, interface_info, passing_global_hiddens={}):
        """
        Forward pass of the model.
        Args:
            batch_input: The input batch.
            interface_info: Information about how to use the model on the dataset. include the following:
                task: The task of the dataset, one of "generation", "multiple_choice_perplexity", "language_model".
                all other keys and values will be passed to the actual interface function as kwargs.
                e.g. max_length in generation, length_normalization in multiple_choice.
            pass_global_hiddens: The global hiddens that are passed to the model. e.g. passing current_step to the model.
        Returns:
            batch_output: The output of the model.
        """
        self._clear_hiddens()
        if self.pass_batch_input:
            self.global_hidden_dict["batch_input"] = batch_input
        if passing_global_hiddens:
            self.global_hidden_dict.update(passing_global_hiddens)

        if self.has_addon("to_device", ""):
            _, batch_input = self.get_addon("to_device", "").pre_forward(**batch_input)

        with torch.autocast(device_type=self._device_type, dtype=self._dtype):
            batch_output = InterfaceMixin.__call__(self, batch_input, interface_info)
        return batch_output

    def save_states(self, checkpoint_path):
        raise NotImplementedError()

    def recover_states(self, checkpoint_path):
        raise NotImplementedError()

    def share_memory(self):
        for param in self.torch_model.parameters():
            param.share_memory_()
        for buffer in self.torch_model.buffers():
            buffer.share_memory_()

    def get_description(self):
        param_counts = self.count_parameters()
        return [
            f"Model class: {self.__class__.__name__}",
            f"Based on torch model: {self.torch_model.__str__()[:200]}...",
            f" and tokenizer: {self.tokenizer.__class__.__name__}",
            f"Addons installed: {[f'{key} x{len(value)}' for key, value in self.addons.items()]}",
            f"Interface selected: {self.interface_dict}",
            f"Parameters: {param_counts[0]:,} ({param_counts[1]:,} trainable + {param_counts[2]:,} frozen)",
        ]

    def named_trainable_parameters(self):
        if getattr(self, "_named_trainable_parameters", None) is None:
            self._named_trainable_parameters = OrderedDict(
                (name, param)
                for name, param in self.torch_model.named_parameters()
                if param.requires_grad
            )
        return self._named_trainable_parameters

    def count_parameters(self):
        num_params = sum(p.numel() for p in self.torch_model.parameters())
        num_trainable_params = sum(
            p.numel() for p in self.named_trainable_parameters().values()
        )
        num_frozen_params = num_params - num_trainable_params
        return num_params, num_trainable_params, num_frozen_params
