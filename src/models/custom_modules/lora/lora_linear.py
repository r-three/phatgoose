import gin
import torch
import torch.nn as nn
import torch.nn.functional as F


@gin.configurable(allowlist=["d_bottleneck", "init_scale"])
class LoRALinear(nn.Module):
    def __init__(self, linear_layer, d_bottleneck, init_scale):
        super().__init__()
        self.d_in = linear_layer.in_features
        self.d_out = linear_layer.out_features
        self.d_bottleneck = d_bottleneck
        self.weight = linear_layer.weight
        self.bias = linear_layer.bias
        self.expert_lora_a = nn.Parameter(
            torch.randn(self.d_bottleneck, linear_layer.in_features) * init_scale
        )
        self.expert_lora_b = nn.Parameter(
            torch.zeros(linear_layer.out_features, self.d_bottleneck)
        )

    def forward(self, input):
        weight = self.weight
        weight = (
            weight
            + torch.matmul(self.expert_lora_b, self.expert_lora_a) / self.d_bottleneck
        )
        return F.linear(input, weight, self.bias)
