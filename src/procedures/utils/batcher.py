# Batcher: takes data config, dataset reader cache (add more as needed), tokenizer. It creates some kind of dataloader, and returns (i) a generator of batches.
import gin
import torch

from src.procedures.utils.sampler import MultiTaskBatchSampler


def create_collate_fn(pad_token_id, max_length=None):
    def collate_fn(batch):
        fields = {field for example in batch for field in example.keys()}
        output_batch = {}
        for key in fields:
            if key in ["input_ids", "target_ids"]:
                output_batch[key] = torch.nn.utils.rnn.pad_sequence(
                    [example[key] for example in batch],
                    batch_first=True,
                    padding_value=pad_token_id,
                )
                if max_length is not None:
                    output_batch[key] = output_batch[key][..., :max_length]
                # cast to long
                output_batch[key] = output_batch[key].long()
            elif key == "answer_choices_ids":
                flat_answer_choice_ids = [
                    choice for example in batch for choice in example[key]
                ]
                num_choice = [len(example[key]) for example in batch]
                if max(num_choice) == 0:
                    continue
                # if max(num_choice) != min(num_choice) or max(num_choice) == 0:
                #     continue
                #     raise NotImplementedError(
                #         "The collate_fn is not implmented for variable number of choices"
                #     )
                if max(num_choice) != min(num_choice):
                    # print(f"Encountered variable number of choices: {num_choice}")
                    padded_choices = []
                    for example in batch:
                        example_choices = example[key]
                        while len(example_choices) < max(num_choice):
                            example_choices.append(torch.tensor([pad_token_id]))
                        padded_choices.extend(example_choices)
                    flat_answer_choice_ids = padded_choices
                flat_answer_choices_ids = torch.nn.utils.rnn.pad_sequence(
                    flat_answer_choice_ids, batch_first=True, padding_value=pad_token_id
                )
                output_batch[key] = flat_answer_choices_ids.view(
                    len(batch), max(num_choice), -1
                )
                if max_length is not None:
                    output_batch[key] = output_batch[key][..., :max_length]
            elif key == "label":
                output_batch[key] = torch.cat([example[key] for example in batch])
            else:
                output_batch[key] = [example[key] for example in batch]

        return output_batch

    return collate_fn


class BaseBatcher(object):
    def __init__(self, shuffle, drop_last, num_workers):
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.seed = None
        self._rng = None

    def set_seed(self, seed):
        self.seed = seed

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def build(self, datasets):
        raise NotImplementedError()


@gin.configurable
class MultiTaskBatcher(BaseBatcher):
    def __init__(
        self,
        shuffle,
        drop_last,
        num_workers,
        temperature,
        num_replicas=-1,
        rank=-1,
        mixing_ratio=None,
    ):
        super().__init__(shuffle, drop_last, num_workers)
        self.temperature = temperature
        self.mixing_ratio = mixing_ratio
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.rank = rank

        assert drop_last, "drop_last must be True for MultiTaskBatcher"

    def build(self, datasets):
        joint_dataset = torch.utils.data.ConcatDataset(datasets)
        max_length = max([dataset.max_length for dataset in datasets])
        # TODO: assuming that all datasets have the same tokenizer
        dataloader = torch.utils.data.DataLoader(
            joint_dataset,
            batch_sampler=MultiTaskBatchSampler(
                dataset_sizes=[len(dataset) for dataset in datasets],
                batch_sizes=[dataset.batch_size for dataset in datasets],
                temperature=self.temperature,
                seed=self.seed,
                shuffle=self.shuffle,
                mixing_ratio=self.mixing_ratio,
                num_replicas=self.num_replicas,
                rank=self.rank,
            ),
            num_workers=self.num_workers,
            collate_fn=create_collate_fn(
                datasets[0].tokenizer.pad_token_id, max_length
            ),
        )
        return dataloader


# https://pytorch.org/docs/stable/notes/randomness.html
@gin.configurable
class SingleTaskBatcher(BaseBatcher):
    def build(self, dataset):
        if isinstance(dataset, list):
            assert len(dataset) == 1, "SingleTaskBatcher only supports one dataset"
            dataset = dataset[0]
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset.batch_size,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
            collate_fn=create_collate_fn(
                dataset.tokenizer.pad_token_id, dataset.max_length
            ),
            generator=generator,
        )
        return data_loader
