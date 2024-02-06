import glob
import os
import re
from typing import List, Optional, Union

import gin
import torch

import src.utils.logging as logging
from src.models.manipulations.architecture_specific import get_model_re_pattern
from src.models.manipulations.device import (
    pipeline_parallelism,
    single_device,
    tensor_parallelism,
)
from src.utils.save_to_gcp import save_to_gcp


@gin.configurable(
    allowlist=["trainable_params", "mix_precision"],
)
def set_trainable_params(
    model,
    trainable_params: Union[str, List[str]],
    mix_precision: Union[str, None] = None,
):
    """
    Set trainable parameters of the model. This is a speical manipulation that the model automatically runs after user-defined manipulations.
    In the case of shared parameters, it is trainable if any of its names matches the trainable pattern.

    Args:
        model: model to manipulate
        trainable_params: trainable parameters
        mix_precision: mix precision type, one of "bf16", "fp16" and None

    """
    if mix_precision == "bf16":
        model._dtype = torch.bfloat16
    elif mix_precision == "fp16":
        model._dtype = torch.float16
    elif mix_precision is None:
        model._dtype = torch.float32
    else:
        raise ValueError(f"Unsupported mix precision type {mix_precision}")

    with torch.no_grad():
        trainable_pattern = get_model_re_pattern(model, trainable_params)

        for param_name, param in model.torch_model.named_parameters():
            param.requires_grad = False
        for param_name, param in model.torch_model.named_parameters():
            if re.fullmatch(trainable_pattern, param_name):
                param.requires_grad = True
        # Move frozen parameters to the specified dtype to save memory
        # While trainble parameters should stay in fp32.
        if mix_precision is not None:
            for param_name, param in model.torch_model.named_parameters():
                if not param.requires_grad and param.data.dtype == torch.float32:
                    param.data = param.data.to(model._dtype)


@gin.configurable(
    allowlist=["device", "parallelism"],
)
def set_device_and_parallelism(
    model,
    device: str = "auto",
    parallelism: Union[str, None] = None,
):
    """Set device and parallelism of the model. (This manipulation is automatically added by the Model class after user-defined manipulations.)

    Args:
        model: model to manipulate
        device: device to use, one of "cpu", "cuda", "cuda:0", "cuda:1", or multiple devices separated by "," (e.g. "cuda:0,cuda:1")
        parallelism: parallelism type, one of None, "pipeline", "module". (Currently only support single process methods.)
    """
    print(f"Setting device to {device} and parallelism to {parallelism}")
    if device == "auto":
        if torch.cuda.is_available():
            devices = [
                f"cuda:{cuda_idx}" for cuda_idx in range(torch.cuda.device_count())
            ]
            model._device_type = "cuda"
        else:
            devices = ["cpu"]
            model._device_type = "cpu"
        print(f"Auto detected devices: {devices}")
    else:
        # TODO: Is _device_type "cuda" here?
        devices = device.split(",")
    if len(devices) > 1:
        assert (
            parallelism is not None
        ), "Parallelism must be enabled when using multiple devices."

    if parallelism is None:
        single_device(model, devices=devices)
    elif parallelism == "pipeline":
        pipeline_parallelism(model, devices=devices)
    elif parallelism == "module":
        tensor_parallelism(model, devices=devices)


def _scan_numbers(prefix, extension):
    found_names = glob.glob(f"{prefix}[0-9]*{extension}")
    found_values = [
        int(name[len(prefix) : len(name) - len(extension)]) for name in found_names
    ]
    if len(found_values) == 0:
        found_values = [-1]
    max_value = max(found_values)
    return max_value, f"{prefix}{max_value}{extension}"


# TODO: Think about parallelism and device
@gin.configurable(
    allowlist=[
        "weight_path",
        "add_index",
        "add_global_step",
        "ignore_frozen_parameters",
        "should_save_to_gcp",
        "save_params",
    ],
)
def save_weights(
    model,
    weight_path: str,
    add_index: bool = False,
    add_global_step: bool = False,
    ignore_frozen_parameters: bool = False,
    should_save_to_gcp: bool = False,
    save_params=None,
):
    assert not (add_index and add_global_step), "Cannot add both index and global step"
    weight_path = os.path.expandvars(weight_path)

    state_dict = model.torch_model.state_dict()
    if save_params is not None:
        param_name_set = set(dict(model.torch_model.named_parameters()).keys())
        for param_name in param_name_set:
            if not re.fullmatch(save_params, param_name):
                del state_dict[param_name]

    elif ignore_frozen_parameters:
        param_name_set = set(dict(model.torch_model.named_parameters()).keys())
        trainable_param_set = set(model.named_trainable_parameters().keys())
        for param_name in param_name_set - trainable_param_set:
            del state_dict[param_name]

    if add_index or add_global_step:
        if "." in weight_path.split("/")[-1]:
            extension = "." + weight_path.split(".")[-1]
            weight_path = weight_path[: -len(extension)]
        else:
            extension = ""

        if add_index:
            index, _ = _scan_numbers(weight_path + ".", extension)
            index += 1
        if add_global_step:
            index = logging.global_step
        weight_path = f"{weight_path}.{index}{extension}"

    print(f"Saving weights to {weight_path}")
    folder_name = os.path.dirname(weight_path)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    torch.save(state_dict, weight_path)
    if should_save_to_gcp:
        save_to_gcp(weight_path)


# TODO: Think about parallelism and device
@gin.configurable(
    allowlist=["weight_path", "scan_numbers", "override_step", "skip_if_not_found"],
)
def load_weights(
    model,
    weight_path: str,
    scan_numbers: bool = False,
    override_step: bool = False,
    skip_if_not_found: bool = False,
):
    weight_path = os.path.expandvars(weight_path)

    if scan_numbers:
        if "." in weight_path.split("/")[-1]:
            extension = "." + weight_path.split(".")[-1]
            weight_path = weight_path[: -len(extension)]
        else:
            extension = ""
        max_value, weight_path = _scan_numbers(weight_path + ".", extension)
        if max_value >= 0 and override_step:
            logging.global_step = max_value

    if os.path.exists(weight_path):
        print(f"Loading weights from {weight_path}")
        state_dict = torch.load(weight_path, map_location=torch.device("cpu"))
        load_result = model.torch_model.load_state_dict(state_dict, strict=False)
        assert (
            len(load_result.unexpected_keys) == 0
        ), f"Load model failed, unexpected keys {load_result.unexpected_keys.__str__()}"
    else:
        if skip_if_not_found:
            print(f"Weight file {weight_path} not found, skip loading.")
        else:
            raise ValueError(f"Weight file {weight_path} not found.")


@gin.configurable(
    allowlist=["model_path", "add_index", "add_global_step"],
)
def save_pretrained(
    model,
    model_path: str,
    add_index: bool = False,
    add_global_step: bool = False,
):
    model_path = os.path.expandvars(model_path)

    if add_index or add_global_step:
        if add_index:
            index, _ = _scan_numbers(model_path + ".", "")
            index += 1
        if add_global_step:
            index = logging.global_step
        model_path = f"{model_path}.{index}"

    print(f"Saving model to {model_path}")
    model.torch_model.save_pretrained(model_path)


@gin.configurable(
    allowlist=["model_path", "scan_numbers", "override_step", "skip_if_not_found"],
)
def load_pretrained(
    model,
    model_path: str,
    scan_numbers: bool = False,
    override_step: bool = False,
    skip_if_not_found: bool = False,
):
    model_path = os.path.expandvars(model_path)

    if scan_numbers:
        max_value, model_path = _scan_numbers(model_path + ".", "")
        if max_value >= 0 and override_step:
            logging.global_step = max_value

    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model_class = model.torch_model.__class__
        model.torch_model = None
        model.torch_model = model_class.from_pretrained(model_path)
    else:
        if skip_if_not_found:
            print(f"Model file {model_path} not found, skip loading.")
        else:
            raise ValueError(f"Model file {model_path} not found.")
