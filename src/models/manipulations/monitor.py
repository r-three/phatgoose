import re
from typing import List, Union

import gin

from src.models.addons import ExposeHidden, PrepareMask
from src.models.manipulations.architecture_specific import get_model_re_pattern


# TODO: Add prepare_mask and expose_hidden classes for decoder generation mode.
# This should work fine for encoder, and decoder under training or multiple choice.
# But it will not work for decoder under generation. Because in generation mode, the model will
# feed one token at a time, and the hidden states will be overwritten by the next token.
# In addition, beam search may alter the identity of input sequences in the batch.
@gin.configurable(
    allowlist=[
        "prepare_mask_modules",
        "expose_hidden_modules",
        "prepare_mask_addon_name",
        "expose_hiddens_addon_name",
    ]
)
def watch_hiddens(
    model,
    prepare_mask_modules: Union[str, List[str]],
    expose_hidden_modules: Union[str, List[str]],
    prepare_mask_addon_name: str = "prepare_mask",
    expose_hiddens_addon_name: str = "expose_hiddens",
):
    """Add prepare_mask and expose_hiddens addons to the model.
    Args:
        model: a model object
        prepare_mask_module: module shortcuts to prepare the attention mask
        expose_hidden_modules: module shortcuts to monitor the hidden states
        prepare_mask_addon_name: name for registering prepare_mask addons
        expose_hiddens_addon_name: name for registering expose_hiddens addons
    """
    # get module dict
    module_dict = model.get_module_dict(exclude_addons=True)

    # translate module shortcuts to re pattern
    prepare_mask_pattern = get_model_re_pattern(model, prepare_mask_modules)
    expose_hiddens_pattern = get_model_re_pattern(model, expose_hidden_modules)

    # add prepare_mask
    if model.has_addon(prepare_mask_addon_name):
        prepare_mask_addon = model.get_addon(prepare_mask_addon_name)
    else:
        matched_modules = [
            module_name
            for module_name in module_dict
            if re.fullmatch(prepare_mask_pattern, module_name)
        ]
        assert (
            len(matched_modules) == 1
        ), f"Only one module should be matched by prepare_mask_modules. However, {len(matched_modules)} are matched."
        module_name = matched_modules[0]
        prepare_mask_addon = PrepareMask(
            global_hidden_dict=model.global_hidden_dict,
            host_module=None,
            write_mask_key=(
                "mask",
                prepare_mask_addon_name,
                module_name,
            ),
        )
        model.insert_addon(
            prepare_mask_addon_name, module_name, prepare_mask_addon, "last"
        )

    # add expose_hiddens
    for module_name, module in module_dict.items():
        if re.fullmatch(expose_hiddens_pattern, module_name):
            if model.has_addon(expose_hiddens_addon_name, module_name):
                continue
            expose_hiddens_addon = ExposeHidden(
                global_hidden_dict=model.global_hidden_dict,
                host_module=module,
                write_hidden_key=(
                    "hidden_states",
                    expose_hiddens_addon_name,
                    module_name,
                ),
                read_mask_key=prepare_mask_addon.write_mask_key,
            )
            model.insert_addon(
                expose_hiddens_addon_name, module_name, expose_hiddens_addon, "last"
            )
