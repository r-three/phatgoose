import json
import os
from collections import OrderedDict

import gin
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

import src.utils.logging as logging
from src.procedures.procedure import Procedure
from src.procedures.utils.batcher import SingleTaskBatcher


@gin.configurable(
    allowlist=[
        "model_name",
        "datasets",
        "include_answer_choices",
        "make_expert_library",
        "expert_library_dir",
        "dataset_length",
    ]
)
# Set the random seed
class Retriever(Procedure):
    linking_fields = ["datasets"]

    def __init__(
        self,
        model_name,
        datasets,
        include_answer_choices,
        make_expert_library,
        expert_library_dir,
        dataset_length,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.datasets = datasets
        self.include_answer_choices = include_answer_choices
        self.make_expert_library = make_expert_library
        self.expert_library_dir = expert_library_dir
        self.dataset_length = dataset_length
        if not os.path.exists(self.expert_library_dir):
            os.makedirs(self.expert_library_dir)

    def link(self):
        if isinstance(self.datasets, str):
            self.datasets = [self.datasets]
        super().link()

    def late_init(self):
        for dataset in self.datasets:
            dataset.set_tokenizer(self.model.tokenizer)
            dataset.peek_stats = {}

    def run(self, step=None):
        logging.print_single_bar()
        print(f"Running {self.name}...")
        self.model.eval()
        with torch.no_grad():
            if self.make_expert_library:
                for dataset in self.datasets:
                    print(f"\t Creating expert library for {dataset.name}...")
                    all_embeddings = []
                    for i in range(len(dataset)):
                        if i >= self.dataset_length:
                            print(f"Stopping at {i}th example")
                            break
                        example = dataset[i]
                        # encode each example in dataset using model, save to file
                        if self.include_answer_choices:
                            if "answer_choices" in example:
                                answer_choices = example["answer_choices"]
                                answer_choice_str = ", ".join(answer_choices)
                            else:
                                answer_choice_str = "None"
                            text = f"Answer Choices: {answer_choice_str}, Instance: {example['input_str']}"
                        else:
                            text = example["input_str"]
                        embedding = self.model.encode(text)
                        all_embeddings.append(embedding)
                    all_embeddings = np.array(all_embeddings)
                    save_file = os.path.join(
                        self.expert_library_dir,
                        f"{dataset.name.replace('/', '_')}_K{self.dataset_length}_embeddings.npy",
                    )
                    np.save(
                        save_file,
                        all_embeddings,
                    )
                    print(f"Saved {dataset.name} to expert library {save_file}\n")
            else:
                train_dataset_embeddings = []
                file_names = []
                # go through all files in self.expert_library_dir and read
                for filename in os.listdir(self.expert_library_dir):
                    if filename.endswith(".npy") and "TRAIN" in filename:
                        file_names.append(filename)
                        with open(
                            os.path.join(self.expert_library_dir, filename), "rb"
                        ) as f:
                            train_dataset_embeddings.append(np.load(f))
                train_dataset_index = []
                for index, each_dataset_embedding in enumerate(
                    train_dataset_embeddings
                ):
                    train_dataset_index.extend(
                        [index] * each_dataset_embedding.shape[0]
                    )
                train_dataset_index = np.array(train_dataset_index)
                train_dataset_embeddings = np.concatenate(
                    train_dataset_embeddings, axis=0
                )
                normalized_train_dataset_embeddings = (
                    train_dataset_embeddings
                    / np.linalg.norm(train_dataset_embeddings, axis=1, keepdims=True)
                )
                print(f"\tDone loading {len(file_names)} files {file_names}\n")
                for dataset in self.datasets:
                    print(
                        f"Retrieving for {dataset.name} from {dataset.name.replace('/', '_')}_K{self.dataset_length}_embeddings.npy..."
                    )
                    eval_dataset_embedding_path = os.path.join(
                        self.expert_library_dir,
                        f"{dataset.name.replace('/', '_')}_K{self.dataset_length}_embeddings.npy",
                    )
                    with open(eval_dataset_embedding_path, "rb") as f:
                        eval_dataset_embeddings = np.load(f)
                    # compute similarity between eval_dataset_embeddings and train_dataset_embeddings
                    normalized_eval_dataset_embeddings = (
                        eval_dataset_embeddings
                        / np.linalg.norm(eval_dataset_embeddings, axis=1, keepdims=True)
                    )
                    similarity = np.dot(
                        normalized_eval_dataset_embeddings,
                        normalized_train_dataset_embeddings.T,
                    )
                    top1_indices = np.argmax(similarity, axis=1)
                    top1_train_dataset_index = train_dataset_index[top1_indices]
                    # take the value from top1_train_dataset_index that occurs the most
                    retrieved_index = np.bincount(top1_train_dataset_index).argmax()
                    # print(f"Retrieved indices are {top1_train_dataset_index}\n")
                    print(f"Retrieved dataset is {file_names[retrieved_index]}\n")

    def save_states(self):
        # TODO(Checkpointing): save results and rng state
        pass

    def recover_states(self):
        # TODO(Checkpointing): load results and rng state
        pass

    def get_description(self):
        return [
            f"Procedure class: {self.__class__.__name__}",
            f"Retrieves using {self.model_name} model on {len(self.datasets)} datasets ({[dataset.name for dataset in self.datasets]})",
        ]
