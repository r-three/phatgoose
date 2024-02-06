import gin
import torch
import torch.nn as nn

from src.models.addons.addon import Addon
from src.models.custom_modules.cpt.lower_level_fn import prutune_tensor_
from src.utils.constants import BOOL_PLACEHOLDER, FLOAT_EPSILON


@gin.configurable()
class PruningGate(Addon):
    has_pre_forward = BOOL_PLACEHOLDER
    pre_forward_returns = BOOL_PLACEHOLDER
    has_post_forward = BOOL_PLACEHOLDER
    post_forward_returns = BOOL_PLACEHOLDER
    _ref_attr_names = ["d_states"]

    def __init__(
        self,
        host_module,
        global_hidden_dict,
        write_dim_key,
        d_states,
        dim=-1,
        position="after",
        gumble_sigmoid=True,
        gumble_stretch=(-1.5, 1.5),
        gumble_epsilon=1e-2,
        gumble_init=3.0,
        pruning_threshold=FLOAT_EPSILON,
    ):
        super().__init__(global_hidden_dict)
        self.write_dim_key = write_dim_key
        self.d_states = d_states
        self.dim = dim
        self.position = position
        self.gumble_sigmoid = gumble_sigmoid
        if self.gumble_sigmoid:
            self.gumble_stretch = gumble_stretch
            self.gumble_epsilon = gumble_epsilon
            self.gumble_init = gumble_init
        self.pruning_threshold = pruning_threshold
        self._resolve_ref_attrs(host_module)

        if self.position == "after":
            self.has_pre_forward = self.pre_forward_returns = False
            self.has_post_forward = self.post_forward_returns = True
        elif self.position == "before":
            self.has_pre_forward = self.pre_forward_returns = True
            self.has_post_forward = self.post_forward_returns = False
        else:
            raise ValueError(f"Invalid position: {self.position}")

        if self.gumble_sigmoid:
            self.bias = nn.Parameter(gumble_init * torch.ones(self.d_states))
            if self.gumble_stretch is not None:
                self.register_buffer(
                    "stretch_b", torch.tensor(self.gumble_stretch[0]), persistent=False
                )
                self.register_buffer(
                    "stretch_w",
                    torch.tensor(self.gumble_stretch[1] - self.gumble_stretch[0]),
                    persistent=False,
                )
                self.register_buffer(
                    "stretch_correction",
                    -torch.tensor(
                        -self.gumble_stretch[0] / self.gumble_stretch[1]
                    ).log(),
                    persistent=False,
                )
            else:
                self.stretch_b = 0
                self.stretch_w = 1
                self.stretch_correction = None
        self.scalar = nn.Parameter(
            (torch.ones(self.d_states) / self.get_gumble_gate(False)).clamp(max=10)
        )

    def get_gumble_gate(self, training=None):
        if self.gumble_sigmoid:
            if training is None:
                training = self.training
            if training:
                uniform = torch.rand(
                    self.scalar.size(0),
                    device=self.scalar.device,
                ).clamp(self.gumble_epsilon, 1 - self.gumble_epsilon)
                offset = torch.log(uniform) - torch.log(1 - uniform)
            else:
                offset = 0
            gate = torch.sigmoid(offset + self.bias)
            gate = (gate * self.stretch_w + self.stretch_b).clamp(0, 1)
            return gate
        else:
            return 1

    def _forward(self, input_states):
        gate = self.scalar
        if self.gumble_sigmoid:
            gate = gate * self.get_gumble_gate()

        expected_activated = torch.sigmoid(self.bias + self.stretch_correction)
        if expected_activated.numel() > 0:
            dim = expected_activated.sum()
        else:
            dim = expected_activated.new_zeros(1).mean()
        self.global_hidden_dict[self.write_dim_key] = dim

        size = [1] * input_states.ndim
        size[self.dim] = -1
        output_states = input_states * gate.view(*size)

        return output_states

    def pre_forward(self, input_states, *args, **kwargs):
        input_states = self._forward(input_states)
        args = [input_states] + list(args)
        return args, kwargs

    def post_forward(self, module_outputs, *args, **kwargs):
        if isinstance(module_outputs, (tuple, list)):
            module_outputs = list(module_outputs)
            module_outputs[0] = self._forward(module_outputs[0])
        else:
            module_outputs = self._forward(module_outputs)
        return module_outputs, args, kwargs

    @torch.no_grad()
    def get_mask(self):
        mask = self.scalar * self.get_gumble_gate(False)
        pruned = mask.abs() < self.pruning_threshold
        mask = mask.masked_fill(pruned, float("nan"))
        return mask


@gin.configurable(
    allowlist=[
        "compute_param_fn",
        "loss_coeff",
        "loss_type",
    ]
)
class PruningLoss(Addon):
    has_post_forward = True
    _ref_attr_names = ["compute_param_fn"]

    def __init__(
        self,
        host_module,
        global_hidden_dict,
        read_dim_keys,
        write_scale_key,
        write_loss_key,
        compute_param_fn="host_module.compute_param_fn",
        loss_coeff=1.0,
        loss_type="log",
    ):
        super().__init__(global_hidden_dict)
        self.read_dim_keys = read_dim_keys
        self.write_scale_key = write_scale_key
        self.write_loss_key = write_loss_key
        self.compute_param_fn = compute_param_fn
        self.loss_coeff = loss_coeff
        self.loss_type = loss_type
        self._resolve_ref_attrs(host_module)
        assert self.loss_type in ["log", "sqrt"]

    def _forward(self, gate_dims):
        scale = self.compute_param_fn(gate_dims)
        if not isinstance(scale, torch.Tensor):
            loss = 0
        if self.loss_type == "log":
            loss = torch.log(scale) * self.loss_coeff
        elif self.loss_type == "sqrt":
            loss = torch.sqrt(scale) * self.loss_coeff
        return scale, loss

    def post_forward(self, *args, **kwargs):
        gate_dims = {}
        for key in self.read_dim_keys:
            if key in self.global_hidden_dict:
                gate_dims[key[-1]] = self.global_hidden_dict[key]
        scale, loss = self._forward(gate_dims)
        self.global_hidden_dict[self.write_scale_key] = scale
        self.global_hidden_dict[self.write_loss_key] = loss
        return args, kwargs


@gin.configurable(allowlist=["update_coeff", "position"])
class HiddenPCA(Addon):
    has_pre_forward = BOOL_PLACEHOLDER
    has_post_forward = BOOL_PLACEHOLDER

    def __init__(
        self,
        host_module,
        global_hidden_dict,
        read_mask_key,
        update_coeff=1e-3,
        position="after",
    ):
        super().__init__(global_hidden_dict)
        self.read_mask_key = read_mask_key
        self.update_coeff = update_coeff
        self.position = position
        assert self.position in ["before", "after"]

        if self.position == "before":
            self.has_pre_forward = True
            self.has_post_forward = False
        elif self.position == "after":
            self.has_post_forward = True
            self.has_pre_forward = False

    def _forward(self, input_states):
        if self.training:
            if self.read_mask_key is not None:
                token_mask = self.global_hidden_dict[self.read_mask_key]
                token_mask = token_mask.bool()
            seq_len, d_states = input_states.size()[-2:]
            if token_mask.size()[-1] != seq_len:
                token_mask = token_mask[..., -seq_len:]
            non_pad_states = (
                input_states.detach()
                .masked_select(token_mask.unsqueeze(-1))
                .view(-1, d_states)
            )
            num_states = non_pad_states.size()[0]
            batch_stats = non_pad_states.T @ non_pad_states / num_states
            if not hasattr(self, "hidden_stats"):
                self.register_buffer(
                    "hidden_stats", batch_stats * self.update_coeff, persistent=True
                )
            else:
                self.hidden_stats = (
                    self.hidden_stats * (1 - self.update_coeff)
                    + batch_stats * self.update_coeff
                )

    def pre_forward(self, hidden_states, *args, **kwargs):
        self._forward(hidden_states)

    def post_forward(self, module_outputs, *args, **kwargs):
        self._forward(module_outputs[0])

    @torch.no_grad()
    def get_permutation(self):
        if not hasattr(self, "hidden_stats"):
            return None
        else:
            return torch.linalg.svd(self.hidden_stats)[0]

    @torch.no_grad()
    def prutune(self, hidden_mask):
        """
        Prune the hidden state stats according to the hidden_mask.
        """
        if hidden_mask is not None and hasattr(self, "hidden_stats"):
            bool_mask = hidden_mask.isnan().logical_not()
            prutune_tensor_(self.hidden_stats, 0, None, bool_mask)
            prutune_tensor_(self.hidden_stats, 1, None, bool_mask)
