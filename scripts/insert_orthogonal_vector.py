import pickle
import re
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--init_orthogonal", action="store_true")
    parser.add_argument("--block_num", default=-1, type=int)
    args = parser.parse_args()
    exp_name = args.exp_name
    exp_path = f"exp_out/{exp_name}"

    # vector in the orthogonal subspace
    def gram_schmidt(vectors):
        basis = []
        for v in vectors:
            w = v - sum(torch.dot(v, u) / torch.dot(u, u) * u for u in basis)
            if torch.norm(w) > 1e-6:  # Avoid adding zero-length vectors
                basis.append(w)
        orthonormal_basis = [u / torch.norm(u + 1e-6) for u in basis]
        return orthonormal_basis

    if args.init_orthogonal:
        print(
            f"For making orthogonal vector, loading from {exp_path}/best_with_averaged_hiddens.pt"
        )
        checkpoint = torch.load(
            f"{exp_path}/best_with_averaged_hiddens.pt", map_location="cpu"
        )
    else:
        print(f"For making orthogonal vector, loading from {exp_path}/best.pt")
        checkpoint = torch.load(f"{exp_path}/best.pt", map_location="cpu")

    regex_pattern = r"(.*)__(\d+)$"
    grouped_values = {}

    for key, value in checkpoint.items():
        if "expert_embeddings" in key:
            match = re.match(regex_pattern, key)
            if match:
                prefix = match.group(1)
                index = match.group(2)
                if prefix not in grouped_values:
                    grouped_values[prefix] = {}
                grouped_values[prefix][index] = value

    for prefix in grouped_values:
        other_vectors = []
        original_vector = None
        for index in grouped_values[prefix]:
            if index == args.index:
                original_vector = grouped_values[prefix][index].float()
            else:
                other_vectors.append(grouped_values[prefix][index].float())
        basis = gram_schmidt(other_vectors)
        orthogonal_subspace_vector = original_vector - sum(
            torch.dot(original_vector, u) / torch.dot(u, u) * u for u in basis
        )
        for index in grouped_values[prefix]:
            key_name = f"{prefix}__{index}"
            if index == args.index:
                pattern = r"encoder\.block\.(\d+)\.layer\.\d+"
                match = re.match(pattern, prefix)
                if match:
                    block_num = int(match.group(1))
                    if block_num <= args.block_num:
                        checkpoint[key_name] = grouped_values[prefix][index]
                        print(f"Skipping orthgonalizing {key_name}")
                    else:
                        checkpoint[key_name] = orthogonal_subspace_vector
                else:
                    checkpoint[key_name] = orthogonal_subspace_vector
            else:
                checkpoint[key_name] = grouped_values[prefix][index]

    if args.init_orthogonal:
        print(
            f"Made Orthogonal, saving to {exp_path}/best_with_averaged_hiddens_ol_init.pt"
        )
        torch.save(checkpoint, f"{exp_path}/best_with_averaged_hiddens_ol_init.pt")
    else:
        print(
            f"Made Orthogonal, saving to {exp_path}/best_with_orthogonal_subspace_vector.pt"
        )
        torch.save(checkpoint, f"{exp_path}/best_with_orthogonal_subspace_vector.pt")
