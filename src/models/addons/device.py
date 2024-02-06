import gin
import torch
import torch.nn as nn

from src.models.addons.addon import Addon


@gin.configurable
class ToDevice(Addon):
    has_pre_forward = True
    pre_forward_returns = True
    _ref_attr_names = []

    def __init__(
        self,
        host_module,
        global_hidden_dict,
    ):
        super().__init__(global_hidden_dict)
        self.device_probe = nn.Parameter(
            torch.Tensor(
                0,
            )
        )

    def pre_forward(self, *args, **kwargs):
        device = self.device_probe.device
        args = [
            arg.to(device) if isinstance(arg, torch.Tensor) else arg for arg in args
        ]
        kwargs = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in kwargs.items()
        }
        return args, kwargs
