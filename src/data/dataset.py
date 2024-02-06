# Dataset: takes data config, read data files, prepare examples and templates, computeMetric_fn.
from collections import defaultdict

import gin
import numpy
import torch

from src.utils import multiprocessing

NUM_PEEK_EXAMPLES = 100


@gin.configurable()
class InterfaceInfo:
    INTERFACE_KEYS = [
        "interface",
        "num_beams",
        "max_gen_length",
        "length_normalization",
        "multiple_choice_loss",
        "unlikelihood_loss",
        "input_loss",
    ]
    """
    An interface is a set of attributes that are required to run a model on a dataset.
    This class contains all the possible optional kwargs that can be passed to the model.
    But, only the ones that are required by the model interface function will be used.

    We require each dataset to have one and only one interface.
    This can be inconvenient in the case of lm in training and gen in evaluation.
    However, since we usually have seperate dataset for train and dev splits, this is not a big issue.
    In the future, we can add some logic to enable sharing underlying data between different datasets.

    Note: max_length in the interface refers to the maximum length of the generated sequence.
    """

    def __init__(
        self,
        interface,
        **kwargs,
    ):
        self.interface = interface
        for key, value in kwargs.items():
            if key not in self.INTERFACE_KEYS:
                print(f"Warning: {key} is not in INTERFACE_KEYS. Please check.")
            setattr(self, key, value)


@gin.configurable(
    allowlist=[
        "dataset_path",
        "split",
        "batch_size",
        "seed",
        "max_examples_per_dataset",
        "metrics",
        "max_length",
    ]
)
class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        name,
        dataset_path,
        split,
        batch_size,
        seed=42,
        max_examples_per_dataset=None,
        metrics=None,
        max_length=None,
    ):
        """
        name: Name of the dataset.
        split: Split of the dataset.
        batch_size: Batch size of the dataset.
        dataset_seed: Seed of the dataset.
        max_examples_per_dataset: Maximum number of examples to use from the dataset. Useful for few-shot learning.
        metrics: Metrics to compute on the dataset.
        max_length: Maximum sequence length when tokenizing the dataset. This affect both input and targt sequences, and their lengths are counted seperately.
        """
        self.name = name
        self.dataset_path = dataset_path
        self.split = split
        self.batch_size = batch_size
        self.seed = seed
        self.max_examples_per_dataset = max_examples_per_dataset
        self.max_length = max_length
        self.metrics = metrics
        self.interface_info = InterfaceInfo()
        self._rng = numpy.random.default_rng(self.seed)

        self.tokenizer = None
        self.load_data()
        self.process_data()
        self.truncate_dataset()

    def set_tokenizer(self, tokenizer):
        if self.tokenizer is None:
            self.tokenizer = tokenizer
            self.peek_examples()
        else:
            assert (
                self.tokenizer.vocab == tokenizer.vocab
            ), "We only support one tokenizer per dataset. "

    def process_data(self):
        """
        This function is meant to handle dataset-specific quirks. So that load_data can be reused as much as possible.
        """
        pass

    def load_data(self):
        if self.dataset_path[0] == "huggingface":
            import pyarrow  # noqa: F401 # to get PowerPC to work
            from datasets import load_dataset as load_huggingface_dataset

            if self.split == "train_validation":
                self._examples = load_huggingface_dataset(*self.dataset_path[1:])
                from datasets import concatenate_datasets

                self._examples = concatenate_datasets(
                    [self._examples["train"], self._examples["validation"]]
                )
            else:
                self._examples = load_huggingface_dataset(
                    *self.dataset_path[1:], split=self.split
                )
            # TODO: it takes a lot of time for FLAN datasets, we can do this inside P3 datasets if needed
            # self._examples = [example for example in self._examples]
        elif self.dataset_path[0] == "datalabs":
            from datalabs import load_dataset as load_datalabs_dataset

            self._examples = load_datalabs_dataset(
                *self.dataset_path[1:], split=self.split
            )
            self._examples = [example for example in self._examples]
        else:
            raise NotImplementedError(
                "If you want to use a custom dataset, you need to implement the load_data method in the child class."
            )

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, idx):
        return {
            "idx": idx,
            "dataset_name": self.name,
        }

    def truncate_dataset(self):
        if self.max_examples_per_dataset is not None:
            self._examples = self._rng.choice(
                self._examples, self.max_examples_per_dataset, replace=False
            ).tolist()

    def share_memory(self):
        shared_list = multiprocessing.get_list_cls()
        self._examples = shared_list(self._examples)

    def get_description(self):
        return [
            f"Dataset class {self.__class__.__name__}",
            f"Size: {len(self)} examples.",
            f"Loaded from {self.split} split of {self.dataset_path}",
            f"Batch size: {self.batch_size}",
            f"Peek stats: {self.peek_stats}",
        ]

    def peek_examples(self):
        num_peek_examples = min(NUM_PEEK_EXAMPLES, len(self))
        if num_peek_examples == 0:
            self.peek_stats = None
        else:
            self.peek_stats = {
                "num_tokens": 0,
                "truncation_ratio": 0,
            }
            for idx in self._rng.choice(
                len(self), num_peek_examples, replace=False
            ).tolist():
                self._peek = defaultdict(int)
                _ = self[idx]
                self.peek_stats["num_tokens"] += (
                    self._peek["num_tokens"] / num_peek_examples
                )
                self.peek_stats["truncation_ratio"] += (
                    self._peek["num_truncated_tokens"]
                    / self._peek["num_full_tokens"]
                    / num_peek_examples
                )
            self.peek_stats["num_tokens"] = int(self.peek_stats["num_tokens"])
            self.peek_stats["truncation_ratio"] = round(
                self.peek_stats["truncation_ratio"], 2
            )

    def tokenize(self, text):
        """
        Tokenize one or multiple text segments, truncate each one and combine them into a single sequence.
        The longer segments are truncated prior to the shorter ones.
        If tokenizer has a bos_token_id and/or eos_token_id, they are prepended and appended respectively.
        """
        tokens = []
        if isinstance(text, str):
            text = [text]
        if self.tokenizer.bos_token_id is not None:
            tokens.append(torch.LongTensor([self.tokenizer.bos_token_id]))
        for segment_idx, segment in enumerate(text):
            segment_tokens = self.tokenizer(
                segment,
                return_tensors="pt",
                truncation=False,
                add_special_tokens=False,
            ).input_ids.flatten()
            tokens.append(segment_tokens)
        if self.tokenizer.eos_token_id is not None:
            tokens.append(torch.LongTensor([self.tokenizer.eos_token_id]))
        lengths = [len(segment_tokens) for segment_tokens in tokens]

        if self.max_length is None or sum(lengths) <= self.max_length:
            per_seg_length = max(lengths)
        else:
            spare_length = self.max_length
            lengths.sort()
            per_seg_length = spare_length // len(lengths)
            for segment_idx, segment_length in enumerate(lengths):
                if per_seg_length > segment_length:
                    spare_length -= segment_length
                    per_seg_length = spare_length // (len(lengths) - segment_idx - 1)
                else:
                    break

        tokens = torch.cat(
            [segment_tokens[:per_seg_length] for segment_tokens in tokens], dim=0
        )
        if hasattr(self, "_peek"):
            self._peek["num_tokens"] += tokens.size(0)
            self._peek["num_full_tokens"] += sum(lengths)
            self._peek["num_truncated_tokens"] += sum(lengths) - tokens.size(0)

        return tokens
