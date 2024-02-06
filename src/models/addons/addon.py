# Addons are similar to hooks, however, they differ by:
# 1. Addons are nn.Module rather than functions, which allows them to be trainable.
# 2. Hooks are meant to be used in debuging or profiling, while addons are long-term modifications to the model.
# 3. Addons are more flexible than hooks, they may have variables that controls their behavior, and the user can decide when to change the variables.
import torch.nn as nn


class Addon(nn.Module):
    has_pre_forward = False
    has_post_forward = False
    pre_forward_returns = False
    post_forward_returns = False
    _ref_attr_names = []

    def __init__(self, global_hidden_dict):
        super().__init__()
        self.global_hidden_dict = global_hidden_dict

    def pre_forward(self, *args, **kwargs):
        raise NotImplementedError("This addon does not support pre forward")

    def post_forward(self, module_outputs, *args, **kwargs):
        raise NotImplementedError("This addon does not support post forward")

    def _resolve_ref_attrs(self, host_module):
        for attr_name in self._ref_attr_names:
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, str):
                if attr_value.startswith("host_module."):
                    setattr(self, attr_name, getattr(host_module, attr_value[12:]))
