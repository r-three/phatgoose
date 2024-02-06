import gin
import torch

from src.models.addons.addon import Addon
from src.utils.constants import BOOL_PLACEHOLDER, FLOAT_EPSILON


@gin.configurable(
    allowlist=[
        "reduction_method",
        "position",
        "mode",
    ]
)
class ExposeHidden(Addon):
    has_pre_forward = BOOL_PLACEHOLDER
    has_post_forward = BOOL_PLACEHOLDER

    def __init__(
        self,
        host_module,
        global_hidden_dict,
        write_hidden_key,
        read_mask_key,
        reduction_method="masked_mean",
        position="before",
        mode="write",
    ):
        super().__init__(global_hidden_dict)
        self.write_hidden_key = write_hidden_key
        self.read_mask_key = read_mask_key
        self.reduction_method = reduction_method
        self.position = position
        self.mode = mode
        assert self.reduction_method in [None, "mean", "masked_mean", "masked_select"]
        assert self.position in ["before", "after"]
        assert self.mode in ["write", "append"]

        if self.position == "before":
            self.has_pre_forward = True
            self.has_post_forward = False
        elif self.position == "after":
            self.has_post_forward = True
            self.has_pre_forward = False

    def _forward(self, hidden_states):
        # hidden_states [batch_size, seq_len, hidden_size]
        # mask [batch_size, seq_len]
        mask = self.global_hidden_dict[self.read_mask_key]
        self.global_hidden_dict[self.read_mask_key] = mask.to(hidden_states.dtype)

        if self.reduction_method == "masked_mean":
            hidden_states = torch.bmm(mask.unsqueeze(-2), hidden_states).squeeze(-2)
        elif self.reduction_method == "masked_select":
            hidden_states = hidden_states.masked_select(
                (mask > FLOAT_EPSILON).unsqueeze(-1)
            ).view(-1, hidden_states.size(-1))
        elif self.reduction_method == "mean":
            hidden_states = hidden_states.mean(dim=1)

        if self.mode == "append":
            if self.write_hidden_key not in self.global_hidden_dict:
                self.global_hidden_dict[self.write_hidden_key] = []
            self.global_hidden_dict[self.write_hidden_key].append(hidden_states)
        elif self.mode == "write":
            self.global_hidden_dict[self.write_hidden_key] = hidden_states

    def pre_forward(self, hidden_states, *args, **kwargs):
        self._forward(hidden_states)

    def post_forward(self, module_outputs, *args, **kwargs):
        # here, we assume that the first output of the module is the hidden states
        # Might need to update this?
        if type(module_outputs) == tuple:
            self._forward(module_outputs[0])
        else:
            self._forward(module_outputs)


@gin.configurable
class PrepareMask(Addon):
    has_pre_forward = True

    def __init__(
        self,
        host_module,
        global_hidden_dict,
        write_mask_key,
    ):
        super().__init__(global_hidden_dict)
        self.write_mask_key = write_mask_key

    def pre_forward(self, *args, **kwargs):
        # attention_mask [batch_size, seq_len]
        input_ids = kwargs["input_ids"]
        attention_mask = kwargs["attention_mask"]
        attention_mask = attention_mask.float()[..., -input_ids.size(-1) :]
        self.global_hidden_dict[self.write_mask_key] = attention_mask / (
            attention_mask.sum(-1, keepdim=True) + FLOAT_EPSILON
        )
