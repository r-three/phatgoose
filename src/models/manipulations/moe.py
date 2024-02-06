import re
from typing import List, Union

import gin

from src.models.addons import FFNExperts, MoELink, Router, ScalerExperts
from src.models.manipulations.architecture_specific import get_model_re_pattern
from src.models.manipulations.utils import search_by_prefix


@gin.configurable
def make_moe(
    model,
    expert_class: str,
    expert_modules: Union[str, List[str]],
    router_modules: Union[str, List[str]],
    expert_addon_name: str = "expert",
    router_addon_name: str = "router",
    expose_hiddens_addon_name: str = "expose_hiddens",
):
    """Adds MoE addons to the model, including expert, router. This function requires the model to have expose_hiddens addons.
    It includes zero experts when created.
    Args:
        model: a model object
        expert_class: the class of the expert, "ffn" or "scaler".
        expert_modules: module shortcuts for adding experts.
        router_modules: module shortcuts for adding routers.
        expert_addon_name: name for registering expert addons.
        router_addon_name: name for registering router addons.
        expose_hiddens_addon_name: name used to register expose_hiddens addons.
    """
    # get module dict
    module_dict = model.get_module_dict(exclude_addons=True)

    # translate module shortcuts to re pattern
    expert_pattern = get_model_re_pattern(model, expert_modules)
    router_pattern = get_model_re_pattern(model, router_modules)

    # add routers
    for module_name, module in module_dict.items():
        if re.fullmatch(router_pattern, module_name):
            if model.has_addon(router_addon_name, module_name):
                continue
            expose_hiddens_addons = model.get_addons(expose_hiddens_addon_name)
            matched_expose_hidden_module_name = search_by_prefix(
                query=module_name,
                target_list=expose_hiddens_addons.keys(),
            )
            matched_expose_hidden_addon = expose_hiddens_addons[
                matched_expose_hidden_module_name
            ]
            router_addon = Router(
                global_hidden_dict=model.global_hidden_dict,
                host_module=module,
                read_hidden_key=matched_expose_hidden_addon.write_hidden_key,
                write_routing_weights_key=(
                    "routing_weights",
                    router_addon_name,
                    module_name,
                ),
                moe_link=MoELink(),
            )
            model.insert_addon(router_addon_name, module_name, router_addon, "last")

    # add experts
    if expert_class == "ffn":
        expert_class = FFNExperts
    elif expert_class == "scaler":
        expert_class = ScalerExperts
    else:
        raise ValueError(f"Unknown expert class {expert_class}")
    for module_name, module in module_dict.items():
        if re.fullmatch(expert_pattern, module_name):
            if model.has_addon(expert_addon_name, module_name):
                continue
            router_addons = model.get_addons(router_addon_name)
            matched_router_module_name = search_by_prefix(
                query=module_name,
                target_list=router_addons.keys(),
            )
            matched_router_addon = router_addons[matched_router_module_name]
            # Note: you can use "host_module.in_features" and "host_module.out_features"
            expert_addon = expert_class(
                global_hidden_dict=model.global_hidden_dict,
                host_module=module,
                read_routing_weights_key=matched_router_addon.write_routing_weights_key,
                moe_link=matched_router_addon.moe_link,
            )
            # Is this a good way?
            if "adapter" in expert_addon_name:
                model.insert_addon(expert_addon_name, module_name, expert_addon, "last")
            elif "lora" in expert_addon_name:
                model.insert_addon(
                    expert_addon_name, module_name, expert_addon, "inner"
                )
            else:
                raise ValueError(f"Unknown expert addon name {expert_addon_name}")


@gin.configurable
def extend_moe(
    model,
    num_new_experts: int,
    weight_init: str,
    router_addon_name: str = "router",
    identifier_stem: str = "",
):
    """Extend the number of experts in the model.
    Args:
        model: a model object
        num_new_experts: number of new experts to add
        weight_init: weight initialization method
        router_addon_name: name used to register router addons.
        identifier_stem: identifier stem for the new experts.
    """
    for router_addon in model.get_addons(router_addon_name).values():
        moe_link = router_addon.moe_link
        moe_link.extend(num_new_experts, identifier_stem)
        moe_link.router.extend(num_new_experts, weight_init)
        for moe_layer in moe_link.moe_layers:
            moe_layer.extend(num_new_experts, weight_init)


@gin.configurable
def make_router_blockwise(
    model,
):
    """Tie the parameters of the routers in the same block."""
    self_attn_modules = {}
    for name, module in model.torch_model.named_modules():
        if "0.SelfAttention._addons.router" in name:
            self_attn_modules[name] = module

    for name, module in model.torch_model.named_modules():
        if "router" in name and "0.SelfAttention._addons.router" not in name:
            new_name = name.replace(
                "1.DenseReluDense._addons.router", "0.SelfAttention._addons.router"
            )
            new_name = new_name.replace(
                "1.EncDecAttention._addons.router", "0.SelfAttention._addons.router"
            )
            new_name = new_name.replace(
                "2.DenseReluDense._addons.router", "0.SelfAttention._addons.router"
            )
            for param in module._parameters:
                module._parameters[param] = self_attn_modules[new_name]._parameters[
                    param
                ]
