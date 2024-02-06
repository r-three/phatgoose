import re
from typing import List, Union

import gin
import torch.nn as nn

from src.models.addons import FFNAdapter, ScalerAdapter
from src.models.custom_modules.lora import LoRALinear
from src.models.manipulations.architecture_specific import get_model_re_pattern


@gin.configurable(
    allowlist=[
        "adapter_class",
        "adapter_modules",
        "adapter_addon_name",
    ]
)
def insert_adapters(
    model,
    adapter_class: str,
    adapter_modules: Union[str, List[str]],
    adapter_addon_name: str,
):
    """
    Insert adapters into the model.
    Args:
        model: model to insert adapters into
        adapter_class: adapter class
        adapter_modules: modules to insert adapters into
        adapter_addon_name: name of the adapter addon
    """
    # get module dict
    module_dict = model.get_module_dict(exclude_addons=True)

    # translate module shortcuts to re pattern
    adapter_pattern = get_model_re_pattern(model, adapter_modules)

    # add adapters
    if adapter_class == "ffn":
        adapter_class = FFNAdapter
    elif adapter_class == "scaler":
        adapter_class = ScalerAdapter
    else:
        raise ValueError(f"Unknown adapter class {adapter_class}")
    for module_name, module in module_dict.items():
        if re.fullmatch(adapter_pattern, module_name):
            if model.has_addon(adapter_addon_name, module_name):
                continue
            adapter_addon = adapter_class(
                global_hidden_dict=model.global_hidden_dict,
                host_module=module,
            )
            model.insert_addon(adapter_addon_name, module_name, adapter_addon, "inner")


@gin.configurable(
    allowlist=[
        "fold_modules",
        "foldable_adapter_addon_name",
    ]
)
def fold_adapters(
    model, fold_modules: Union[str, List[str]], foldable_adapter_addon_name: str
):
    """
    Fold adapter addons into their host modules.
    Args:
        model: model to perform folding
        fold_modules: modules to fold adapters into
        foldable_adapter_addon_name: name of the adapter addon
    """
    # get module dict
    module_dict = model.get_module_dict(exclude_addons=True)

    # translate module shortcuts to re pattern
    fold_pattern = get_model_re_pattern(model, fold_modules)

    # fold peft addons
    for module_name, module in module_dict.items():
        if re.fullmatch(fold_pattern, module_name):
            addon = model.get_addon(foldable_adapter_addon_name, module_name)
            addon.fold(module)
    model.remove_addon(foldable_adapter_addon_name)


@gin.configurable(allowlist=["lora_modules", "lora_layers"])
def modify_with_lora(
    model, lora_modules: Union[str, List[str]], lora_layers: Union[str, List[str]]
):
    lora_modules = get_model_re_pattern(model, lora_modules)
    lora_layers = get_model_re_pattern(model, lora_layers)
    for module_name, module in dict(model.torch_model.named_modules()).items():
        if re.fullmatch(lora_modules, module_name):
            for c_name, layer in dict(module.named_children()).items():
                if re.fullmatch(lora_layers, c_name):
                    assert isinstance(
                        layer, nn.Linear
                    ), f"LoRA can only be applied to torch.nn.Linear, but {layer} is {type(layer)}."
                    setattr(
                        module,
                        c_name,
                        LoRALinear(layer),
                    )
