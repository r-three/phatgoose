import functools
from collections import OrderedDict, defaultdict

import torch.nn as nn

from src.models.addons import Addon


class AddonHostMixin(nn.Module):
    """
    AddonHostMixin is a mixin that allow to attach addons to a module.
    Addons acts similar to hooks, but they are nn.Module rather than functions.
    Before adding any addon, we need to use AddonHostMixin to augment an existing module object.
    Overall, this design serves to manipulate a pretrained model while avoiding copying specific model implementations for other packages.

    An addon can have pre and post forword calls, the first one take place before the host module forward pass, and the second one take place after the host module forward pass.
    One host module can have multiple addons, but each addon need to have a unique name.
    Since multiple addon calls will happen sequtially, when inserting one more addon to a module with existing addons, you will need to speificy the call order:
        call_order "first" use both pre&post_forward_call before other addons
        call_order "last" use both pre&post_forward_call after other addons
        call_order "inner" use pre_forward_call after other addons and post_forward_call before other addons
        call_order "outer" use pre_forward_call before other addons and post_forward_call after other addons

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_addons()

    @classmethod
    def augment_object(cls, object):
        object.__class__ = cls
        object._init_addons()

    def _init_addons(self):
        self._addons = nn.ModuleDict()
        self._pre_forward_calls = []
        self._post_forward_calls = []

    def insert_addon(self, addon_name, addon_module, call_order):
        if addon_name in self._addons:
            raise ValueError(f"Addon {addon_name} already exists in {self}.")
        self._addons[addon_name] = addon_module
        if addon_module.has_pre_forward:
            if call_order == "first" or call_order == "outer":
                self._pre_forward_calls.insert(0, addon_name)
            elif call_order == "last" or call_order == "inner":
                self._pre_forward_calls.append(addon_name)
            else:
                raise ValueError(f"Unknown call order {call_order}.")

        if addon_module.has_post_forward:
            if call_order == "first" or call_order == "inner":
                self._post_forward_calls.insert(0, addon_name)
            elif call_order == "last" or call_order == "outer":
                self._post_forward_calls.append(addon_name)
            else:
                raise ValueError(f"Unknown call order {call_order}.")

    def remove_addon(self, addon_name):
        if addon_name not in self._addons:
            raise ValueError(f"Addon {addon_name} does not exist in {self}.")
        addon = self._addons.pop(addon_name)
        if addon.has_pre_forward:
            self._pre_forward_calls.remove(addon_name)
        if addon.has_post_forward:
            self._post_forward_calls.remove(addon_name)

    def forward(self, *args, **kwargs):
        for addon_name in self._pre_forward_calls:
            addon = self._addons[addon_name]
            if addon.pre_forward_returns:
                args, kwargs = addon.pre_forward(*args, **kwargs)
            else:
                addon.pre_forward(*args, **kwargs)
        module_outputs = super().forward(*args, **kwargs)
        for addon_name in self._post_forward_calls:
            addon = self._addons[addon_name]
            if addon.post_forward_returns:
                module_outputs, args, kwargs = addon.post_forward(
                    module_outputs, *args, **kwargs
                )
            else:
                addon.post_forward(module_outputs, *args, **kwargs)
        return module_outputs


@functools.lru_cache(maxsize=None)
def get_augmented_class(InputClass):
    if issubclass(InputClass, AddonHostMixin):
        return InputClass

    class AugmentedClass(AddonHostMixin, InputClass):
        pass

    return AugmentedClass


def augment_module(module):
    if not isinstance(module, AddonHostMixin):
        AugmentedClass = get_augmented_class(module.__class__)
        AugmentedClass.augment_object(module)


class AddonControlMixin:
    """
    AddonControlMixin is a mixin to support controlling addons.
    This differs from AddonHostMixin in that AddonHostMixin is used on the submodules,
    while AddonControlMixin is used on the top-level model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.addons = OrderedDict()
        self._module_dict = self._module_dict_no_addons = None

    def has_addon(self, addon_name, module_name=None):
        """
        Check if an addon exists in (a particular module of) the model.
        Args:
            addon_name: The name of the addon.
            module_name: The name of the module that the addon is attached to.
        Returns:
            True if the addon exists, False otherwise.
        """
        if module_name is None:
            return addon_name in self.addons
        else:
            return addon_name in self.addons and module_name in self.addons[addon_name]

    def insert_addon(self, addon_name, module_name, addon, call_order):
        """
        Insert an addon into the model.
        Args:
            addon_name: The name of the addon.
            module_name: The name of the module that the addon is attached to.
            addon: The addon module.
            call_order: The order of the addon in the forward pass, one of "first", "last", "outer", "inner".
        """
        if self.has_addon(addon_name, module_name):
            raise ValueError(
                f"Addon {addon_name} already exists in {module_name}. Use a different addon name or reusing the existing addon."
            )
        if addon_name not in self.addons:
            self.addons[addon_name] = OrderedDict()
        self.addons[addon_name][module_name] = addon
        module = self.get_module_dict()[module_name]
        augment_module(module)
        module.insert_addon(addon_name, addon, call_order)
        self._named_trainable_parameters = None
        self._module_dict = None

    def remove_addon(self, addon_name, module_name=None):
        """
        Remove an addon from the model.
        Args:
            addon_name: The name of the addon.
            module_name: The name of the module that the addon is attached to.
        """
        if module_name is None:
            to_remove = list(self.addons[addon_name])
        else:
            to_remove = [module_name]
        for module_name in to_remove:
            module = self.get_module_dict()[module_name]
            module.remove_addon(addon_name)
            del self.addons[addon_name][module_name]
        if len(self.addons[addon_name]) == 0:
            del self.addons[addon_name]
        self._named_trainable_parameters = None
        self._module_dict = None

    def get_addons(self, addon_name):
        """
        Retrieve a group of addons with the same name.
        Args:
            addon_name: The name of the addon.
        Returns:
            A dictionary of addons with the same name, keyed by the module name.
        """
        return self.addons[addon_name]

    def get_addon(self, addon_name, module_name=None):
        """
        Retrieve an addon that is the only one with the given name.
        Args:
            addon_name: The name of the addon.
            module_name: The name of the module that the addon is attached to.
        Returns:
            The addon under the given addon_name.
        """
        if module_name is not None:
            return self.addons[addon_name][module_name]
        else:
            addon_found = None
            for addon in self.addons[addon_name].values():
                if addon_found is None:
                    addon_found = addon
                else:
                    assert (
                        addon is addon_found
                    ), "More than one addon found. Please specify the module_name."
            return addon_found

    def get_module_dict(self, exclude_addons=False):
        """
        Get the module_dict, this is helpful for a function that traverse and modify the model.
        Args:
            exclude_addons: Whether to exclude addons from the module_dict.
        Returns:
            The module_dict of the model.
        """
        key = "_module_dict_no_addons" if exclude_addons else "_module_dict"
        if getattr(self, key, None) is None:
            if exclude_addons:
                self._module_dict_no_addons = OrderedDict(
                    (name, module)
                    for name, module in self.torch_model.named_modules()
                    if not isinstance(module, Addon)
                )
            else:
                self._module_dict = OrderedDict(self.torch_model.named_modules())
        return getattr(self, key)
