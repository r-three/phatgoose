"""Implements a distributed sampler to sample different tasks with
temperature sampling in a way to make sure that the same task is
selected in each core. Modified from Hyperformer codebase."""
from typing import List, Optional, TypeVar

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler

T_co = TypeVar("T_co", covariant=True)


class MultiTaskBatchSampler(Sampler[T_co]):
    """Defines a sampler to sample multiple datasets with temperature sampling
    in a distributed fashion."""

    def __init__(
        self,
        dataset_sizes: List[int],
        batch_sizes: List[int],
        temperature: float,
        shuffle: bool = True,
        num_replicas: Optional[int] = -1,
        rank: Optional[int] = -1,
        seed: int = 42,
        mixing_ratio: Optional[List[float]] = None,
    ) -> None:
        """Constructor for MultiTaskBatchSampler.
        Args:
            dataset_sizes: a list of integers, specifies the number of samples in
                each dataset.
            batch_sizes: a list of integer, specifies the batch size in each dataset.
            temperature: float, temperature used for temperature sampling. The larger
                the value, the datasets are sampled equally, and for value of 0, the datasets
                will be sampled according to their number of samples.
            num_replicas: integer, specifies the number of processes.
            rank: integer, specifies the rank of the current process.
            seed: integer, random seed.
            shuffle: bool, if set to true, the datasets will be shuffled in each epoch.
        """
        if num_replicas == -1:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank == -1:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1)
            )
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_sizes = batch_sizes
        self.dataset_sizes = dataset_sizes
        # By default we drop the last elements if dataset is not divisible by the number of ranks.
        self.rank_dataset_sizes = [
            dataset_size // self.num_replicas for dataset_size in self.dataset_sizes
        ]
        self.dataset_offsets = torch.cumsum(torch.LongTensor([0] + dataset_sizes), 0)
        self.total_sizes = [
            dataset_size
            // (self.num_replicas * batch_size)
            * (self.num_replicas * batch_size)
            for dataset_size, batch_size in zip(self.dataset_sizes, self.batch_sizes)
        ]
        self.temperature = temperature
        self.seed = seed
        self.epoch = 0
        self.num_batches_per_epoch = sum(
            [
                dataset_size // (self.num_replicas * batch_size)
                for dataset_size, batch_size in zip(
                    self.dataset_sizes, self.batch_sizes
                )
            ]
        )
        self.shuffle = shuffle
        self.mixing_ratio = mixing_ratio
        self.batch_size = min(self.batch_sizes)

    def generate_tasks_distribution(self):
        """Given the dataset sizes computes the weights to sample each dataset
        according to the temperature sampling."""
        if self.mixing_ratio is not None:
            assert len(self.mixing_ratio) == len(
                self.dataset_sizes
            ), f"Size mismatch between mixing ratio {len(self.mixing_ratio)} and number of datasets: {self.dataset_sizes}"
            return torch.as_tensor(self.mixing_ratio, dtype=torch.double)

        total_size = sum(self.dataset_sizes)
        weights = np.array(
            [
                (size / total_size) ** (1.0 / self.temperature)
                for size in self.dataset_sizes
            ]
        )
        weights = weights / np.sum(weights)
        return torch.as_tensor(weights, dtype=torch.double)

    def __iter__(self):
        # print(f"the value of epoch is {self.epoch}")
        # Defines torch generator, to make random choices consistent across cores in
        # different epochs, the seed needs to be set based on seed and epoch.
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        # Shuffles the datasets if shuffle is set to true, and shards the datasets per rank.
        self.rank_indices = []
        for dataset_size, total_size in zip(self.dataset_sizes, self.total_sizes):
            if self.shuffle:
                dataset_indices = torch.randperm(
                    dataset_size, generator=generator
                ).tolist()
            else:
                dataset_indices = list(range(dataset_size))
            self.rank_indices.append(
                dataset_indices[self.rank : total_size : self.num_replicas]
            )

        # To make the model consistent across different processes, since the
        # model is based on tasks, we need to make sure the same task is selected
        # across different processes.
        tasks_distribution: torch.Tensor = self.generate_tasks_distribution()

        # Chooses the tasks which will be used in each batch in one epoch.
        # With passing generator, we make sure this choice is consistent across
        # different processes.
        batch_task_assignments = torch.multinomial(
            tasks_distribution,
            self.num_batches_per_epoch,
            replacement=True,
            generator=generator,
        )
        pointers = [0 for i in range(len(self.rank_indices))]
        # print(f"the batch_task_assignments are {batch_task_assignments}")
        for batch_task in batch_task_assignments:
            # Gets the number of samples of the selected datasets available for the
            # current rank.
            batch_size = self.batch_sizes[batch_task]
            if pointers[batch_task] >= len(self.rank_indices[batch_task]):
                # shuffle the list self.rank_indices[batch_task]
                # print(f"reshuffling indices are {self.rank_indices[batch_task]}")
                self.rank_indices[batch_task] = [
                    self.rank_indices[batch_task][i]
                    for i in torch.randperm(len(self.rank_indices[batch_task]))
                ]
                # print(f"shuffled indices are {self.rank_indices[batch_task]}")
                pointers[batch_task] = 0
            # samples are already randomized in self.rank_indices
            results = (
                self.dataset_offsets[batch_task]
                + torch.tensor(
                    self.rank_indices[batch_task][
                        pointers[batch_task] : pointers[batch_task] + batch_size
                    ]
                )
            ).tolist()
            pointers[batch_task] += batch_size
            # # update self.rank_indices
            # self.rank_indices[batch_task] = (
            #     self.rank_indices[batch_task][batch_size:]
            #     + self.rank_indices[batch_task][:batch_size]
            # )

            # num_task_samples = self.rank_dataset_sizes[batch_task]
            # # Computes the random samples from the chosen dataset.
            # indices = torch.randint(
            #     low=0,
            #     high=num_task_samples,
            #     size=(batch_size,),
            #     generator=generator,
            # ).tolist()
            # # Converts the selected indices to the global indices on the given dataset.
            # results = (
            #     self.dataset_offsets[batch_task]
            #     + torch.tensor(self.rank_indices[batch_task])[indices]
            # ).tolist()
            yield results

        # update self.epoch to have different random samples in the next epoch
        self.epoch += 1

    def __len__(self):
        return self.num_batches_per_epoch

    def set_epoch(self, epoch):
        self.epoch = epoch
        # TODO: Find a way to make DDP work without explicitly setting the epoch.
        # What is an epoch when we have temperature > 1? This feels really weird.
