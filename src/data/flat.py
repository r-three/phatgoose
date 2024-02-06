import json
import os

import gin
import numpy as np

from src.data.dataset import Dataset


@gin.configurable
class FlatDataset(Dataset):
    def __init__(self, input_field, target_field, **kwargs):
        super().__init__(**kwargs)
        self.input_field = input_field
        self.target_field = target_field

    def __getitem__(self, idx):
        example = self._examples[idx]
        input_str = example[self.input_field]
        target_str = example[self.target_field]
        input_ids = self.tokenize(input_str)
        target_ids = self.tokenize(target_str)
        tokenized_example = {
            "example_idx": idx,
            "input_str": input_str,
            "target_str": target_str,
            "input_ids": input_ids,
            "target_ids": target_ids,
            "references": example.get("references", []),
        }
        # add additional keys to tokenized_example
        tokenized_example.update(super().__getitem__(idx))
        tokenized_example.update({f"_{key}": value for key, value in example.items()})

        return tokenized_example


@gin.configurable
class UnnaturalInstructionsDataset(FlatDataset):
    def process_data(self):
        nested_examples = self._examples
        self._examples = []
        for example in nested_examples:
            self._examples.extend(example["instances"])
            if "reformulations" in example:
                self._examples.extend(example["reformulations"])


@gin.configurable
class AlpacaDataset(FlatDataset):
    def load_data(self):
        with open(f"{self.data_path}", "r") as f:
            self._examples = json.load(f)

    def process_data(self):
        for example in self._examples:
            if example["input"] == "":
                example["combined_input"] = (
                    "Below is an instruction that describes a task. Write a response that appropriately completes the request."
                    + "\n\n"
                    + "Instruction:\n"
                    + example["instruction"]
                    + "\n\n"
                    + "Response:\n"
                )
            else:
                example["combined_input"] = (
                    "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
                    + "\n\n"
                    + "Instruction:\n"
                    + example["instruction"]
                    + "\n\n"
                    + "Input:\n"
                    + example["input"]
                    + "\n\n"
                    + "Response:\n"
                )


@gin.configurable
class RedditWritingPromptsDataset(FlatDataset):
    def load_data(self):
        with open(f"{self.data_path}/{self.split}.wp_source", "r") as f:
            input_strs = [line.strip() for line in f]
        with open(f"{self.data_path}/{self.split}.wp_target", "r") as f:
            target_strs = [line.strip() for line in f]
        self._examples = [
            {"input": input_str, "target": target_str}
            for input_str, target_str in zip(input_strs, target_strs)
        ]


@gin.configurable
class DataBricksDolly15kDataset(FlatDataset):
    def process_data(self):
        for example in self._examples:
            if example["context"] == "":
                example["instruction"] = (
                    "Instruction:\n" + example["instruction"] + "\n\n" + "Response:\n"
                )
            else:
                example["instruction"] = (
                    "Context:\n"
                    + example["context"]
                    + "\n\n"
                    + "Instruction:\n"
                    + example["instruction"]
                    + "\n\n"
                    + "Response:\n"
                )


@gin.configurable
class SelfInstructDataset(FlatDataset):
    def process_data(self):
        for example in self._examples:
            if example["input"] == "":
                example["instruction_with_input"] = (
                    "Instruction: \n" + example["instruction"]
                )
            else:
                example["instruction_with_input"] = (
                    "Instruction: \n"
                    + example["instruction"]
                    + "\n\n"
                    + "Input: \n"
                    + example["input"]
                )


@gin.configurable
class UnPredicTable5kDataset(FlatDataset):
    # TODO: Later change the code so that make few shot should be a helper method and can be used any where
    def __init__(self, num_shot, **kwargs):
        self.num_shot = num_shot
        super().__init__(**kwargs)

    def process_data(self):
        new_ds = {}
        for example in self._examples:
            if example["task"] in new_ds:
                new_ds[example["task"]].append(example)
            else:
                new_ds[example["task"]] = [example]
        processed_examples = []

        def process_example(example):
            if len(example["options"]) != 0:
                options_str = ",".join(
                    ["".join(option) for option in example["options"]]
                )
                return (
                    "Input:\n" + example["input"] + "\n" + "Options:\n" + options_str,
                    "Output:\n" + example["output"],
                )
            return "Input:\n" + example["input"], "Output:\n" + example["output"]

        for task in new_ds:
            task_examples = new_ds[task]
            self._rng.shuffle(task_examples)
            context_examples = task_examples[: self.num_shot]
            remaining_examples = task_examples[self.num_shot :]
            context_str = ""
            for example in context_examples:
                inp_str, out_str = process_example(example)
                context_str += inp_str + "\n" + out_str + "\n\n"
            for example in remaining_examples:
                inp_str, _ = process_example(example)
                inp_str += "\n" + "Output:\n"
                out_str = example["output"]
                processed_examples.append(
                    {"input": context_str + inp_str, "output": out_str}
                )
        self._examples = processed_examples


@gin.configurable
class C4Dataset(FlatDataset):
    def load_data(self):
        from datasets import load_dataset as load_huggingface_dataset

        data_files = {"validation": "en/c4-validation.*.json.gz"}
        self._examples = load_huggingface_dataset(
            self.dataset_path, data_files=data_files, split=self.split
        )


@gin.configurable
class FlanDataset(FlatDataset):
    def __init__(self, is_few_shot=False, answer_options="all", **kwargs):
        self.is_few_shot = is_few_shot
        self.answer_options = answer_options
        super().__init__(**kwargs)

    def truncate_dataset(self):
        if (
            self.max_examples_per_dataset is not None
            and self.max_examples_per_dataset < len(self._examples)
        ):
            selected_list = self._rng.choice(
                len(self._examples), size=self.max_examples_per_dataset, replace=False
            )
            self._examples = self._examples.select(selected_list)

    def load_data(self):
        from datasets import load_from_disk

        datasets_offline = (
            os.environ.get("MM_ROOT")
            + "/src/datasets_offline/"
            + "/".join(self.dataset_path)
        )
        if os.path.exists(datasets_offline):
            # complete dataset is stored as train split
            self._examples = load_from_disk(datasets_offline)["train"]
        else:
            raise ValueError(
                f"Offline dataset not found, please download from gcloud. Path not found: {datasets_offline}"
            )

    def process_data(self):
        if self.answer_options == "opt":
            if self.is_few_shot:
                self._examples = self._examples.filter(
                    lambda example: "fs_opt" in example["_template_type"]
                )
            else:
                self._examples = self._examples.filter(
                    lambda example: "zs_opt" in example["_template_type"]
                )
        elif self.answer_options == "noopt":
            if self.is_few_shot:
                self._examples = self._examples.filter(
                    lambda example: "fs_noopt" in example["_template_type"]
                )
            else:
                self._examples = self._examples.filter(
                    lambda example: "zs_noopt" in example["_template_type"]
                )
        elif self.answer_options == "all":
            if self.is_few_shot:
                self._examples = self._examples.filter(
                    lambda example: "fs_noopt" in example["_template_type"]
                    or "fs_opt" in example["_template_type"]
                )
            else:
                self._examples = self._examples.filter(
                    lambda example: "zs_noopt" in example["_template_type"]
                    or "zs_opt" in example["_template_type"]
                )
        else:
            raise ValueError(
                f"answer_options {self.answer_options} is not valid. Use opt, noopt, all."
            )
        self._examples = self._examples.filter(
            lambda example: example["inputs"].strip() != ""
            and example["targets"].strip() != ""
        )
        self._examples = self._examples.map(
            lambda example: {"references": example["targets"]}
        )
        rng = np.random.RandomState(1234)
        selected_list = rng.choice(
            len(self._examples), size=len(self._examples), replace=False
        )
        self._examples = self._examples.select(selected_list)
        num_eval_examples = min(int(len(self._examples) * 0.1), 10_000)
        if self.split == "validation":
            self._examples = self._examples.select(range(num_eval_examples))
        else:
            self._examples = self._examples.select(
                range(num_eval_examples, len(self._examples))
            )


if __name__ == "__main__":
    from src.utils.gin import build

    gin_config = """
    D/UNPREDICTABLE5K/UnPredicTable5kDataset:
        dataset_path = ["huggingface", "MicPie/unpredictable_5k"]
        input_field = "input"
        target_field = "output"
        max_length = 128
        num_shot = 5

    D/UNPREDICTABLE5K/TRAIN/build.cls = @UnPredicTable5kDataset
    D/UNPREDICTABLE5K/TRAIN/UnPredicTable5kDataset:
        split = "train"
        batch_size = 1
    """
    gin.parse_config(gin_config)
    with gin.config_scope("D/UNPREDICTABLE5K/TRAIN"):
        unpredictable_dataset = build(scope_name="D/UNPREDICTABLE5K/TRAIN")
    import ipdb

    ipdb.set_trace()
