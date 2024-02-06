import uuid

import gin
import numpy as np
import torch
import torch.nn as nn
from transformers.activations import ACT2FN

from src.models.addons.addon import Addon
from src.utils.constants import BOOL_PLACEHOLDER


@gin.configurable(
    allowlist=[
        "d_in",
        "d_out",
        "d_bottleneck",
        "non_linearity",
        "position",
        "residual_connection",
        "dimension_compensation",
        "divide_by_d_bottleneck",
    ],
)
class FFNAdapter(Addon):
    """
    This module is a 2-layer FFN adpater module.
    When setting non_linearity to identity, we can also use it to implement LoRA.
    """

    has_pre_forward = BOOL_PLACEHOLDER
    has_post_forward = BOOL_PLACEHOLDER
    pre_forward_returns = BOOL_PLACEHOLDER
    post_forward_returns = BOOL_PLACEHOLDER
    _ref_attr_names = ["d_in", "d_out"]

    def __init__(
        self,
        host_module,
        global_hidden_dict,
        d_in,
        d_out,
        d_bottleneck,
        non_linearity,
        position="beside",
        residual_connection=False,
        dimension_compensation=False,
        divide_by_d_bottleneck=False,
    ):
        """
        Args:
            global_hidden_dict: dict, global hidden states visible to all addon modules
            read_routing_weights_key: str, key to read routing weights ex: ("module_name", "routing_weights")
            d_in: int, input dimension
            d_out: int, output dimension
            d_bottleneck: int, dimension of the bottleneck layer
            non_linearity: str, activation function
            position: str, position of the moe module, can be "before" "beside" or "after"
            residual_connection: bool, whether to use residual connection (helpful for adapters)
            dimension_compensation: bool, whether to divide the output by sqrt(d_bottleneck)
        """
        super().__init__(global_hidden_dict)
        self.d_in = d_in
        self.d_out = d_out
        self.d_bottleneck = d_bottleneck
        self.non_linearity = non_linearity
        self.position = position
        self.residual_connection = residual_connection
        self.dimension_compensation = dimension_compensation
        self.divide_by_d_bottleneck = divide_by_d_bottleneck
        self._resolve_ref_attrs(host_module)
        assert self.position in ["before", "beside", "after"]
        assert not self.residual_connection or self.d_in == self.d_out

        if self.position == "before":
            self.has_pre_forward = self.pre_forward_returns = True
            self.has_post_forward = self.post_forward_returns = False
        elif self.position == "beside":
            self.has_pre_forward = True
            self.pre_forward_returns = False
            self.has_post_forward = self.post_forward_returns = True
            self._temp_hidden_key = None
            assert (
                not residual_connection
            ), "Best not to use residual connection for beside adapters."
        elif self.position == "after":
            self.has_pre_forward = self.pre_forward_returns = False
            self.has_post_forward = self.post_forward_returns = True
        if self.non_linearity == "identity":
            self.activation_fn = lambda x: x
        else:
            self.activation_fn = ACT2FN[non_linearity]

        self.layer1 = nn.Parameter(torch.randn(self.d_in, self.d_bottleneck) * 0.01)
        self.layer2 = nn.Parameter(torch.zeros(self.d_bottleneck, self.d_out))

    def _forward(self, input_hidden):
        """
        Args:
            input_hidden: (..., seq_len, d_in)
        Returns:
            output_hidden: (..., seq_len, d_out)
        """
        mid_hidden = self.activation_fn(input_hidden @ self.layer1)
        if self.dimension_compensation:
            mid_hidden = mid_hidden / np.sqrt(self.d_bottleneck)
        output_hidden = mid_hidden @ self.layer2
        if self.residual_connection:
            output_hidden = output_hidden + input_hidden
        if self.divide_by_d_bottleneck:
            output_hidden = output_hidden / self.d_bottleneck
        return output_hidden

    def pre_forward(self, hidden_states, *args, **kwargs):
        output_hidden = self._forward(hidden_states)
        if self.position == "beside":
            while (
                self._temp_hidden_key is None
                or self._temp_hidden_key in self.global_hidden_dict
            ):
                self._temp_hidden_key = str(uuid.uuid4())[:8]
            self.global_hidden_dict[self._temp_hidden_key] = output_hidden
        elif self.position == "before":
            args = (output_hidden,) + args
            return args, kwargs

    def post_forward(self, module_outputs, *args, **kwargs):
        if self.position == "beside":
            output_hidden = self.global_hidden_dict[self._temp_hidden_key]
            del self.global_hidden_dict[self._temp_hidden_key]
            if isinstance(module_outputs, tuple):
                return (
                    (module_outputs[0] + output_hidden,) + module_outputs[1:],
                    args,
                    kwargs,
                )
            else:
                return module_outputs + output_hidden, args, kwargs
        elif self.position == "after":
            if isinstance(module_outputs, tuple):
                output_hidden = self._forward(module_outputs[0])
                return (output_hidden,) + module_outputs[1:], args, kwargs
            else:
                output_hidden = self._forward(module_outputs)
                return output_hidden, args, kwargs

    @torch.no_grad()
    def fold(self, module):
        assert self.non_linearity == "identity", "Only identity adapters can be folded."
        if not isinstance(module, nn.Linear):
            raise NotImplementedError
        weight = self.layer1 @ self.layer2  # (d_in, d_out)
        if self.dimension_compensation:
            weight = weight / np.sqrt(self.d_bottleneck)
        if self.residual_connection:
            weight = weight + torch.eye(self.d_in)
        if self.position == "beside":
            module.weight.data += weight.T
        elif self.position == "before":
            module.weight.data = module.weight @ weight.T
        elif self.position == "after":
            module.weight.data = weight.T @ module.weight
            if module.bias is not None:
                module.bias.data = weight.T @ module.bias


@gin.configurable(allowlist=["d_hidden", "position"])
class ScalerAdapter(Addon):
    """
    This module scales the features of the input_hidden, and it can be used to implement IA3.
    """

    has_pre_forward = BOOL_PLACEHOLDER
    pre_forward_returns = BOOL_PLACEHOLDER
    has_post_forward = BOOL_PLACEHOLDER
    post_forward_returns = BOOL_PLACEHOLDER
    _ref_attr_names = ["d_hidden"]

    def __init__(self, host_module, global_hidden_dict, d_hidden, position="before"):
        """
        Args:
            global_hidden_dict: dict, global hidden states visible to all addon modules
            d_hidden: int, hidden dimension
        """
        super().__init__(global_hidden_dict)
        self.d_hidden = d_hidden
        self.position = position
        self._resolve_ref_attrs(host_module)
        # if self.
        assert self.position in ["before", "after"]

        if self.position == "before":
            self.has_pre_forward = self.pre_forward_returns = True
            self.has_post_forward = self.post_forward_returns = False
        elif self.position == "after":
            self.has_pre_forward = self.pre_forward_returns = False
            self.has_post_forward = self.post_forward_returns = True

        self.scaler = nn.Parameter(torch.ones(self.d_hidden))

    def _forward(self, input_hidden):
        """
        Args:
            input_hidden: (..., seq_len, d_hidden)
        Returns:
            output_hidden: (..., seq_len, d_hidden)
        """
        output_hidden = input_hidden * self.scaler
        return output_hidden

    def post_forward(self, module_outputs, *args, **kwargs):
        hidden_states = self._forward(module_outputs)
        return hidden_states, args, kwargs

    def pre_forward(self, hidden_states, *args, **kwargs):
        hidden_states = self._forward(hidden_states)
        args = (hidden_states,) + args
        return args, kwargs

    @torch.no_grad()
    def fold(self, module):
        if self.position == "before":
            module.weight.data = module.weight * self.scaler.unsqueeze(0)
        elif self.position == "after":
            module.weight.data = module.weight * self.scaler.unsqueeze(1)
            if module.bias is not None:
                module.bias.data = module.bias * self.scaler


if __name__ == "__main__":
    import copy

    layer = nn.Linear(100, 50)
    before = FFNAdapter(
        layer,
        {},
        100,
        100,
        8,
        non_linearity="identity",
        position="before",
        residual_connection=True,
    )
    beside = FFNAdapter(
        layer,
        {},
        100,
        50,
        8,
        non_linearity="identity",
        position="beside",
        residual_connection=False,
    )
    after = FFNAdapter(
        layer,
        {},
        50,
        50,
        8,
        non_linearity="identity",
        position="after",
        residual_connection=True,
    )
    with torch.no_grad():
        for module in [before, beside, after]:
            for p in module.parameters():
                p.normal_(std=0.01)
    x = torch.randn(10, 100)
    y_before = layer(before._forward(x))
    y_beside = layer(x) + beside._forward(x)
    y_after = after._forward(layer(x))
    layer_before = copy.deepcopy(layer)
    before.fold(layer_before)
    layer_beside = copy.deepcopy(layer)
    beside.fold(layer_beside)
    layer_after = copy.deepcopy(layer)
    after.fold(layer_after)
    assert torch.allclose(layer_before(x), y_before, rtol=1e-4, atol=1e-6)
    assert torch.allclose(layer_beside(x), y_beside, rtol=1e-4, atol=1e-6)
    assert torch.allclose(layer_after(x), y_after, rtol=1e-4, atol=1e-6)
