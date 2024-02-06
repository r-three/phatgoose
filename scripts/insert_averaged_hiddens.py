import os
import pickle
from argparse import ArgumentParser

import numpy as np
import torch

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--orthogonal_constraint", action="store_true")
    parser.add_argument("--out_path")
    parser.add_argument("--hiddens_suffix", required=True)
    parser.add_argument("--component", default=None)
    args = parser.parse_args()
    exp_name = args.exp_name
    dataset_name = args.dataset_name
    index = args.index
    dataset_name = dataset_name.replace("/", "_")
    exp_path = f"exp_out/{exp_name}"
    if args.orthogonal_constraint:
        print(
            f"For averaging hiddens, loading from {exp_path}/best_with_orthogonal_subspace_vector.pt"
        )
        checkpoint = torch.load(
            f"{exp_path}/best_with_orthogonal_subspace_vector.pt", map_location="cpu"
        )
    else:
        print(f"For averaging hiddens, loading from {exp_path}/best.pt")
        checkpoint = torch.load(f"{exp_path}/best.pt", map_location="cpu")
    averaged_hiddens_path = (
        f"{exp_path}/averaged_hiddens{args.hiddens_suffix}/{dataset_name}.pickle"
    )
    print(f"Loading averaged hiddens from {averaged_hiddens_path}")
    with open(averaged_hiddens_path, "rb") as f:
        averaged_hiddens = pickle.load(f)

    if args.component is not None:
        if args.component == "pretrained" or args.component == "task":
            pretrained_hiddens_path = f"exp_out/P3C4_lora/averaged_hiddens{args.hiddens_suffix}/D_C4_EVAL.pickle"
            with open(pretrained_hiddens_path, "rb") as f:
                pretrained_hiddens = pickle.load(f)
            if args.component == "pretrained":
                print(f"Removing task component from averaged hiddens")
                for key in averaged_hiddens:
                    averaged_hiddens[key] = (
                        np.dot(averaged_hiddens[key], pretrained_hiddens[key])
                        / np.dot(pretrained_hiddens[key], pretrained_hiddens[key])
                        * pretrained_hiddens[key]
                    )
            elif args.component == "task":
                print(f"Removing pretrained component from averaged hiddens")
                for key in averaged_hiddens:
                    averaged_hiddens[key] = (
                        averaged_hiddens[key]
                        - np.dot(averaged_hiddens[key], pretrained_hiddens[key])
                        / np.dot(pretrained_hiddens[key], pretrained_hiddens[key])
                        * pretrained_hiddens[key]
                    )
        elif args.component == "specific_task":
            init_hiddens_path = f"{exp_path}/averaged_init_hiddens${hiddens_suffix}/{dataset_name}.pickle"
            with open(init_hiddens_path, "rb") as f:
                init_hiddens = pickle.load(f)
            print(f"Removing init component from averaged hiddens")
            for key in averaged_hiddens:
                averaged_hiddens[key] = (
                    averaged_hiddens[key]
                    - np.dot(averaged_hiddens[key], init_hiddens[key])
                    / np.dot(init_hiddens[key], init_hiddens[key])
                    * init_hiddens[key]
                )
    new_checkpoint = checkpoint.copy()
    if "encoder.final_layer_norm" in averaged_hiddens:
        for key in averaged_hiddens:
            if "encoder" in key and "final_layer_norm" not in key:
                checkpoint_key = f"{key}._addons.router.expert_embeddings__{index}"
                new_checkpoint[checkpoint_key] = torch.tensor(averaged_hiddens[key])
        for key in new_checkpoint:
            if "decoder" in key and "expert_embeddings" in key and key.endswith(str(0)):
                # replace __0 with __{index}
                checkpoint_key = key.replace("__0", f"__{index}")
                averaged_hiddens_key = "encoder.final_layer_norm"
                new_checkpoint[checkpoint_key] = torch.tensor(
                    averaged_hiddens[averaged_hiddens_key]
                )
    else:
        for key in averaged_hiddens:
            checkpoint_key = f"{key}._addons.router.expert_embeddings__{index}"
            new_checkpoint[checkpoint_key] = torch.tensor(averaged_hiddens[key])

    if args.out_path is None:
        out_path = exp_path
    else:
        out_path = args.out_path
        os.makedirs(out_path, exist_ok=True)
    if index == "0":
        print(f"saving averaged hiddens to {out_path}/best.pt")
        torch.save(new_checkpoint, f"{out_path}/best.pt")
    else:
        print(f"saving averaged hiddens to {out_path}/best_with_averaged_hiddens.pt")
        torch.save(new_checkpoint, f"{out_path}/best_with_averaged_hiddens.pt")
