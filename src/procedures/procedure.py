import os

import gin

from src.utils.gin import get_scope_defined_objects


@gin.configurable(allowlist=["seed"])
class Procedure:
    linking_fields = []

    def __init__(self, name, seed=42):
        self.name = name
        self.seed = seed

    def link(self):
        for attr in self.linking_fields:
            linking_names = getattr(self, attr)
            if linking_names is None:
                continue
            elif isinstance(linking_names, str):
                linking_pointers = get_scope_defined_objects(linking_names)
            elif isinstance(linking_names, list):
                linking_pointers = [
                    get_scope_defined_objects(linking_name)
                    for linking_name in linking_names
                ]
            elif isinstance(linking_names, dict):
                linking_pointers = {
                    key: get_scope_defined_objects(linking_name)
                    for key, linking_name in linking_names.items()
                }
            else:
                raise ValueError(
                    f"Linking attribute {attr} has an invalid type {type(linking_names)}."
                )
            setattr(self, attr, linking_pointers)

    def run(self):
        raise NotImplementedError()

    def save_states(self, checkpoint_path):
        raise NotImplementedError()

    def recover_states(self, checkpoint_path):
        raise NotImplementedError()

    def share_memory(self):
        pass

    def late_init(self):
        """
        This is called after all procedures are linked to prepare things that depend on other objects.
        And, we can't do this inside link because it needs to run within the correct gin scope,
        while gin.config_scope doesn't seem to support recursive entry.
        """
        pass
