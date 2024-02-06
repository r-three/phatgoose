import os
import re
from collections import defaultdict

import gin

try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False

try:
    import tensorboardX

    tensorboard_available = True
except ImportError:
    tensorboard_available = False

backend = None
global_step = 0
screen_width = 88


def print_single_bar():
    print("-" * screen_width)


def print_double_bar():
    print("=" * screen_width)


def print_plate(text):
    print("#" * screen_width)
    start_idx = screen_width // 2 - len(text) // 2
    end_idx = start_idx + len(text)
    center_text = (
        "#" + " " * (start_idx - 1) + text + " " * (screen_width - end_idx - 1) + "#"
    )
    print(center_text)
    print("#" * screen_width)


@gin.configurable(
    allowlist=[
        "wandb_project_name",
    ]
)
def wandb_logger_setup(
    exp_name,
    logging_dir,
    starting_step,
    wandb_project_name="${WANDB_PROJECT}",
):
    assert wandb_available, "wandb is not installed."
    global backend
    backend = "wandb"

    wandb_project_name = os.path.expandvars(wandb_project_name)
    logging_dir = os.path.expandvars(logging_dir)
    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    wandb.init(project=wandb_project_name, name=exp_name, dir=logging_dir)
    # TODO: find a way to set starting_step in wandb


def tensorboard_logger_setup(
    exp_name,
    logging_dir,
    starting_step,
):
    assert tensorboard_available, "tensorboardX is not installed."
    global backend
    backend = "tensorboard"

    logging_dir = os.path.expandvars(logging_dir)
    logdir = os.path.join(logging_dir, exp_name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    global tensorboard_writer
    tensorboard_writer = tensorboardX.SummaryWriter(
        logdir=logdir,
        # purge_step=starting_step,
    )
    # TODO: find a way to handle override starting_step in load model or checkpoint


@gin.configurable(
    allowlist=[
        "logging_dir",
        "starting_step",
    ]
)
def logger_setup(
    backend,
    exp_name,
    logging_dir="loggings",
    starting_step=0,
):
    assert (
        exp_name != ""
    ), "exp_name must not be empty. Otherwise, you might lose all previous logs."
    if backend == "wandb":
        wandb_logger_setup(exp_name, logging_dir, starting_step)
    elif backend == "tensorboard":
        tensorboard_logger_setup(exp_name, logging_dir, starting_step)
    elif backend is None:
        pass
    else:
        raise ValueError(f"Unknown backend: {backend}")

    global global_step
    global_step = starting_step


def logger_close():
    if backend == "wandb":
        wandb.finish()
    elif backend == "tensorboard":
        tensorboard_writer.export_scalars_to_json(
            os.path.join(tensorboard_writer.logdir, "scalars.json")
        )
        tensorboard_writer.close()


def logger_step():
    global global_step
    global_step += 1


def flatten_dict(d, parent_key="", sep="/"):
    """Flatten a dictionary.
    Args:
        d (dict): Dictionary to flatten.
        parent_key (str): Parent key.
        sep (str): Separator for keys.
    Returns:
        dict: Flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def log_scalar_dict(scalar_dict):
    scalar_dict = flatten_dict(scalar_dict)
    if backend == "wandb":
        wandb.log(scalar_dict, step=global_step)
    elif backend == "tensorboard":
        for key, value in scalar_dict.items():
            tensorboard_writer.add_scalar(key, value, global_step)


def get_numberless_patterns(list_of_strings):
    """Get a set of numberless patterns from a list of strings.
    Args:
        list_of_strings: List of strings.
    Returns:
        pattern_count: Numberless patterns and their counts.
    """
    pattern_dict = defaultdict(int)
    for string in list_of_strings:
        pattern_dict[re.sub(r"\d+", "*", string)] += 1
    pattern_count_list = list(pattern_dict.items())
    pattern_count_list.sort(key=lambda x: x[1], reverse=True)
    return pattern_count_list


def log_strings_by_pattern(prefix, list_of_strings):
    """Log a list of strings by their numberless patterns.
    Args:
        prefix (str): Prefix to print before the list of strings.
        list_of_strings (list[str]): List of strings.
    """
    print(prefix)
    tabs = "\t" * (prefix.count("\t") + 1)
    pattern_count_list = get_numberless_patterns(list_of_strings)
    for pattern, count in pattern_count_list:
        print(f"{tabs}{pattern}: {count}")


# TODO: Find a way to make the important metrics stand out.
def log_metric_dict(prefix, metric_dict):
    """Log a dictionary of metrics.
    Args:
        prefix (str): Prefix to print before the dictionary.
        metric_dict (dict): Dictionary of metrics.
    """
    print(prefix)
    tabs = "\t" * (prefix.count("\t") + 1)
    for key, value in metric_dict.items():
        print(f"{tabs}{key}: {value}")
