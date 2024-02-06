import os
import re
from argparse import ArgumentParser

import gin
import torch


@gin.configurable
def svd_orth(W):
    norms = torch.norm(W, dim=-1, keepdim=True)
    W_unit = W / norms
    U, S, V = torch.svd(W_unit)
    W_orth_unit = U @ V.T
    W_orth = W_orth_unit * norms
    return W_orth


@gin.configurable
def check_order_svd_orth():
    W1 = torch.rand(5, 10)
    layer_norm = torch.nn.LayerNorm(10)
    W1 = layer_norm(W1)
    W1_orth = svd_orth(W1)

    # Get the shuffled indices
    shuffled_indices = torch.randperm(W1.size(0))

    W2 = W1[shuffled_indices]
    W2_orth = svd_orth(W2)
    # Check if the same shuffling is applied to W1_orth to get W2_orth
    assert torch.allclose(
        W2_orth, W1_orth[shuffled_indices], atol=1e-6
    ), "Ordering mismatch after SVD orthogonalization."


@gin.configurable
def check_redundancy_svd_orth():
    W = torch.tensor([[1.0, 0, 0.0], [0, 1.0, 0.0], [1, 1, 0]])
    W_orth = svd_orth(W)
    import ipdb

    ipdb.set_trace()


@gin.configurable
def check_svd_orth():
    W = torch.rand(5, 10)
    layer_norm = torch.nn.LayerNorm(10)
    W = layer_norm(W)
    W_orth = svd_orth(W)
    import ipdb

    ipdb.set_trace()


@gin.configurable
def make_checkpoint_svd_orth(checkpoint_path, skip_keys=[]):
    checkpoint = torch.load(f"{checkpoint_path}/best.pt")

    regex_pattern = r"(.*)__(\d+)$"
    grouped_values = {}

    for key, value in checkpoint.items():
        if "expert_embeddings" in key:
            match = re.match(regex_pattern, key)
            if match:
                prefix = match.group(1)
                index = match.group(2)
                index = int(index)
                if prefix not in grouped_values:
                    grouped_values[prefix] = {}
                grouped_values[prefix][index] = value
    for prefix in grouped_values:
        W = torch.stack(
            [grouped_values[prefix][index] for index in sorted(grouped_values[prefix])]
        )
        if prefix in skip_keys:
            print(f"Skipping {prefix}")
            W_orth = W
        else:
            W_orth = svd_orth(W)
        for index in sorted(grouped_values[prefix]):
            grouped_values[prefix][index] = W_orth[index]
    for prefix in grouped_values:
        for index in sorted(grouped_values[prefix]):
            checkpoint[f"{prefix}__{index}"] = grouped_values[prefix][index]

    if "lora" in checkpoint_path:
        out_path = checkpoint_path.replace("lora", "lora_svd_orth")
        if len(skip_keys) > 0:
            out_path = out_path.replace("lora", f"lora_skip{len(skip_keys)}")
    elif "adapter" in checkpoint_path:
        out_path = checkpoint_path.replace("adapter", "adapter_svd_orth")
        if len(skip_keys) > 0:
            out_path = out_path.replace("adapter", f"adapter_skip{len(skip_keys)}")
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    print(f"Saving checkpoint to {out_path}")
    torch.save(checkpoint, f"{out_path}/best.pt")


@gin.configurable
def func_caller(func):
    func()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gin_bindings", nargs="+", default=[])
    args = parser.parse_args()
    gin.parse_config(args.gin_bindings)
    func_caller()
