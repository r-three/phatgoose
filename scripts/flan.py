import argparse
from datasets import load_dataset


def filter_and_save_dataset(task_name):
    dataset = load_dataset("Open-Orca/FLAN", cache_dir='~/.cache/huggingface/datasets')['train']

    subset = dataset.filter(
        lambda example: task_name == example['_task_name'],
        load_from_cache_file=False
    )

    subset.save_to_disk(task_name)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process datasets.')
    parser.add_argument('-task_name', type=str, required=True, help='The task name to filter the dataset')

    args = parser.parse_args()

    if not args.task_name:
        raise ValueError("No task_name provided. Please specify a task_name using '-task_name'.")

    filter_and_save_dataset(args.task_name)