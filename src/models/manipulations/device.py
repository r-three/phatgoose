import re
from typing import List, Union

import gin
from transformers.utils.model_parallel_utils import get_device_map

from src.models.addons import ToDevice
from src.models.manipulations.architecture_specific import get_model_re_pattern


def incremental_to(module, device):
    """Move the cpu parameters and buffers of a module to device. Keep the gpu parameters and buffers unchanged.

    Args:
        module: a module
        device: a device
    """
    for param in module.parameters():
        if param.device.type == "cpu":
            param.data = param.data.to(device)
    for buffer in module.buffers():
        if buffer.device.type == "cpu":
            buffer.data = buffer.data.to(device)


@gin.configurable(
    allowlist=[
        "devices",
        "to_device_modules",
        "to_device_addon_name",
        "device_mapping_strategy",
    ]
)
def make_device_adaptive(
    model,
    devices: List[str],
    to_device_modules: Union[str, List[str]],
    to_device_addon_name: str = "to_device",
    device_mapping_strategy: str = "chunk",
):
    """Add ToDevice addons to the model. The addon will be prepended to the addon list, so that it will be called before the module's other addons.
    Args:
        model: a model object
        devices: a list of devices to move the model to.
        to_device_modules: module shortcuts for adding ToDevice addons.
        to_device_addon_name: name for registering ToDevice addons.
        device_mapping_strategy: device mapping strategy, one of "chunk", "round_robin"
    """
    # get module dict
    module_dict = model.get_module_dict(exclude_addons=True)

    # translate module shortcuts to re pattern
    to_device_pattern = get_model_re_pattern(model, to_device_modules)

    # find matched modules and compute device mapping
    matched_modules = []
    for module_name, module in module_dict.items():
        if re.fullmatch(to_device_pattern, module_name):
            matched_modules.append((module_name, module))
    if device_mapping_strategy == "chunk":
        hf_device_mapping = get_device_map(len(matched_modules), devices)
        device_mapping = {
            module_idx: device
            for device, module_idx_list in hf_device_mapping.items()
            for module_idx in module_idx_list
        }
    elif device_mapping_strategy == "round_robin":
        device_mapping = {
            module_idx: devices[module_idx % len(devices)]
            for module_idx in range(len(matched_modules))
        }

    # add addons and move to device
    for module_idx, (module_name, module) in enumerate(matched_modules):
        if model.has_addon(to_device_addon_name, module_name):
            pass
        else:
            to_device_addon = ToDevice(
                global_hidden_dict=model.global_hidden_dict,
                host_module=module,
            )
            model.insert_addon(
                to_device_addon_name, module_name, to_device_addon, "first"
            )

        incremental_to(module, device_mapping[module_idx])


@gin.configurable(allowlist=["devices"])
def single_device(model, devices):
    make_device_adaptive(
        model,
        devices=devices,
        to_device_modules=["model"],
        to_device_addon_name="to_device",
    )


@gin.configurable(allowlist=["devices", "repeated_modules"])
def pipeline_parallelism(
    model, devices, repeated_modules=["encoder_block", "decoder_block"]
):
    # move the repeated modules to seperate devices
    make_device_adaptive(
        model,
        devices=devices,
        to_device_modules=repeated_modules,
        to_device_addon_name="to_device",
        device_mapping_strategy="chunk",
    )
    # move the remaining modules to the first device
    make_device_adaptive(
        model,
        devices=devices[0],
        to_device_modules="model",
        to_device_addon_name="to_device",
    )


@gin.configurable
def tensor_parallelism(
    model,
    devices,
):
    """TODO: Apply tensor parallelism to a model."""
