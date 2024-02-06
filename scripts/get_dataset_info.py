import json
import statistics
from argparse import ArgumentParser
from statistics import mean

import gin
import numpy as np
import torch
from scipy.stats import iqr
from tqdm import tqdm
from transformers import AutoTokenizer

import src.data.p3
from src.utils.gin import build


def get_median(list_of_numbers):
    """

    Args:
        all_scores: list of dictionaries, where one of the value is the score we are interested in

    Returns:

    """
    return round(statistics.median(list_of_numbers), 3)


def get_interquartilerange(list_of_numbers):
    """


    Args:
        list_of_numbers:

    Returns:

    """
    return round(iqr(list_of_numbers), 3)


def get_average(list_of_numbers):
    """

    Args:
        list_of_numbers:

    Returns:

    """
    return round(mean(list_of_numbers), 3)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--gin_files",
        nargs="+",
        default=[
            "colm/datasets/p3_t5xl.gin",
            "colm/datasets/flanv2_t5xl.gin",
            "colm/datasets/bigbench.gin",
        ],
    )
    parser.add_argument("--gin_bindings", nargs="+", default=[])
    parser.add_argument("--dataset_names", required=True, nargs="+")
    args = parser.parse_args()
    gin.parse_config_files_and_bindings(args.gin_files, args.gin_bindings)
    tokenizer = AutoTokenizer.from_pretrained("google/t5-large-lm-adapt")
    for dataset_name in args.dataset_names:
        print(f"Dataset is {dataset_name}")
        with gin.config_scope(dataset_name):
            dataset = build(scope_name=dataset_name)
        dataset.set_tokenizer(tokenizer)
        import ipdb

        ipdb.set_trace()
        len_inputs, len_targets = [], []
        for example in tqdm(dataset):
            len_inputs.append(example["input_ids"].shape[0])
            len_targets.append(example["target_ids"].shape[0])

        # Bucket the len_targets in orders of 10 and give counts for each like histogram

        print(f"Average length of inputs: {get_average(len_inputs)}")
        print(f"Median length of inputs: {get_median(len_inputs)}")
        print(
            f"Interquartile range of length of inputs: {get_interquartilerange(len_inputs)}"
        )
        print(f"Min and Max length of inputs: {min(len_inputs)} and {max(len_inputs)}")

        print(f"Average length of targets: {get_average(len_targets)}")
        print(f"Median length of targets: {get_median(len_targets)}")
        print(
            f"Interquartile range of length of targets: {get_interquartilerange(len_targets)}"
        )
        print(
            f"Min and Max length of targets: {min(len_targets)} and {max(len_targets)}"
        )
