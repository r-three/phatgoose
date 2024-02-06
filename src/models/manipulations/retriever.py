import re
from typing import List, Union

import gin

from src.models.addons import FeatureExtractor
from src.models.manipulations.architecture_specific import get_model_re_pattern


@gin.configurable
def insert_feature_extractor(
    model,
    feature_extractor_module: str = "encoder",
    feature_extractor_addon_name: str = "feature_extractor",
):
    # get module dict
    module_dict = model.get_module_dict(exclude_addons=True)

    # translate module shortcuts to re pattern
    feature_extractor_pattern = get_model_re_pattern(model, feature_extractor_module)

    # add prepare_mask
    if model.has_addon(feature_extractor_addon_name):
        feature_extractor_addon = model.get_addon(feature_extractor_addon_name)
    else:
        matched_modules = [
            module_name
            for module_name in module_dict
            if re.fullmatch(feature_extractor_pattern, module_name)
        ]
        assert (
            len(matched_modules) == 1
        ), f"Only one module should be matched by feature_extractor_module. However, {len(matched_modules)} are matched."
        module_name = matched_modules[0]
        feature_extractor_addon = FeatureExtractor(
            global_hidden_dict=model.global_hidden_dict,
            host_module=None,
            write_hidden_key=(
                "hidden_states",
                feature_extractor_addon_name,
                module_name,
            ),
        )
        model.insert_addon(
            feature_extractor_addon_name, module_name, feature_extractor_addon, "last"
        )
