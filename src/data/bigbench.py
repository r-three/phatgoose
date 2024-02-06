import json
import os

import gin
import numpy as np
import torch

from src.data.dataset import Dataset


def find_label(target, answer_choices):
    if len(answer_choices) == 0:
        return -1
    else:
        matched_len = [
            (
                (target[: len(choice)] == choice[: len(target)]).long().sum(),
                -len(choice),
                idx,
            )
            for idx, choice in enumerate(answer_choices)
        ]
        return max(matched_len)[2]


@gin.configurable
class BigBenchDataset(Dataset):
    def process_data(self):
        # example["targets"] is always a list
        if "exact_match_multiple_ans" in self.metrics:
            self._examples = self._examples.map(
                lambda example: {"references": example["targets"]}
            )
        elif "exact_match" in self.metrics or "rouge" in self.metrics:
            self._examples = self._examples.map(
                lambda example: {"references": example["targets"][0]}
            )
        self._examples = self._examples.map(
            lambda example: {"input": example["inputs"] + "\nAnswer:"}
        )
        self._examples = self._examples.map(
            lambda example: {"target": max(example["targets"], key=len)}
        )

    def truncate_dataset(self):
        if (
            self.max_examples_per_dataset is not None
            and self.max_examples_per_dataset < len(self._examples)
        ):
            selected_list = self._rng.choice(
                len(self._examples), size=self.max_examples_per_dataset, replace=False
            )
            self._examples = self._examples.select(selected_list)

    def __getitem__(self, idx):
        example = self._examples[idx]
        input_str = example["input"]
        if input_str == "":
            input_str = "<NO INPUT>"
        target_str = example["target"]
        answer_choices = example["multiple_choice_targets"]
        input_ids = self.tokenize(input_str)
        target_ids = self.tokenize(target_str)
        answer_choices_ids = [
            self.tokenize(answer_choice) for answer_choice in answer_choices
        ]
        label = find_label(target_ids, answer_choices_ids)
        multi_label = [
            i for i, x in enumerate(example["multiple_choice_scores"]) if x == 1
        ]
        # if len(multi_label) > 1:
        #     print(f"Make sure to use multi label evaluation for {self.name}")
        if len(multi_label) > 0 and label not in multi_label:
            print(f"Please double check the example: {example}")
            print(
                f"label {label} not in multi_label {multi_label} for target_ids {target_ids} and answer_choices_ids {answer_choices_ids}"
            )
        label = torch.LongTensor([label])
        tokenized_example = {
            "example_idx": idx,
            "input_str": input_str,
            "target_str": target_str,
            "input_ids": input_ids,
            "target_ids": target_ids,
            "answer_choices_ids": answer_choices_ids,
            "answer_choices": answer_choices,
            "references": example.get("references", []),
            "label": label,
            "multi_label": multi_label,
        }
        # add additional keys to tokenized_example
        tokenized_example.update(super().__getitem__(idx))
        tokenized_example.update({f"_{key}": value for key, value in example.items()})

        return tokenized_example


@gin.configurable
class BigBenchSampleDataset(Dataset):
    def __init__(self, answer_choices=[], **kwargs):
        self.answer_choices = answer_choices
        super().__init__(**kwargs)

    def process_data(self):
        self._examples = self._examples.map(
            lambda example: {"input": example["input"] + "\nAnswer:"}
        )
        self._examples = self._examples.map(
            lambda example: {"references": example["target"]}
        )

    def truncate_dataset(self):
        if (
            self.max_examples_per_dataset is not None
            and self.max_examples_per_dataset < len(self._examples)
        ):
            selected_list = self._rng.choice(
                len(self._examples), size=self.max_examples_per_dataset, replace=False
            )
            self._examples = self._examples.select(selected_list)

    def __getitem__(self, idx):
        example = self._examples[idx]
        input_str = example["input"]
        target_str = example["target"]
        answer_choices = self.answer_choices
        input_ids = self.tokenize(input_str)
        target_ids = self.tokenize(target_str)
        answer_choices_ids = [
            self.tokenize(answer_choice) for answer_choice in answer_choices
        ]
        label = find_label(target_ids, answer_choices_ids)
        label = torch.LongTensor([label])
        tokenized_example = {
            "example_idx": idx,
            "input_str": input_str,
            "target_str": target_str,
            "input_ids": input_ids,
            "target_ids": target_ids,
            "answer_choices_ids": answer_choices_ids,
            "answer_choices": answer_choices,
            "references": example.get("references", []),
            "label": label,
        }
        # add additional keys to tokenized_example
        tokenized_example.update(super().__getitem__(idx))
        tokenized_example.update({f"_{key}": value for key, value in example.items()})

        return tokenized_example


# @gin.configurable
# class BBReasoningAboutColoredObjects(BigBenchDataset):
#     def process_data(self):
#         from num2words import num2words
#         def textify_number(input_list):
#             update_list = [num2words[x] if x.isdigit() else x for x in input_list]
#             unique_list = []
#             for x in update_list:
#                 if x not in unique_list:
#                     unique_list.append(x)
#             return unique_list
#         self._examples = self._examples.map(
#             lambda example: {"references": textify_number(example["targets"])}
#         )
#         self._examples = self._examples.map(
#             lambda example: {"input": example["inputs"]}
#         )
#         self._examples = self._examples.map(
#             lambda example: {"target": max(example["references"], key=len)}
#         )
