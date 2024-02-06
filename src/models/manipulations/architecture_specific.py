# Handle different model architectures.
# Here we provide a set of model shortcuts and map them to regular expressions.
# The regular expressions are used to find the models to be manipulated.
# This allows us to easily extend this code to other model architectures as well as shortcuts.
import functools
from collections import defaultdict
from typing import List, Union

from transformers import PreTrainedModel

all_model_shortcut_dict = defaultdict(dict)

all_model_shortcut_dict["shared"] = {
    "all": r".*",
    "model": "",
    "nothing": r"$^",
    "expert": r".*expert.*",
}

all_model_shortcut_dict["t5"] = {
    "encoder": r"^encoder$",
    "encoder_block": r"^encoder\.block\.\d+$",
    "encoder_sublayer": r"^encoder\.block\.\d+\.layer\.\d+\.(SelfAttention|DenseReluDense)$",
    "encoder_linear": r"^encoder\.block\.\d+\.layer\.\d+\.(SelfAttention|DenseReluDense)\.(q|k|v|o|wi_\d|wo)$",
    "encoder_final_ln": r"^encoder.final_layer_norm$",
    "decoder": r"^decoder$",
    "decoder_block": r"^decoder\.block\.\d+$",
    "decoder_sublayer": r"decoder\.block\.\d+\.layer\.\d+\.(SelfAttention|EncDecAttention|DenseReluDense)$",
    "decoder_linear": r"^decoder\.block\.\d+\.layer\.\d+\.(SelfAttention|EncDecAttention|DenseReluDense)\.(q|k|v|o|wi_\d|wo)$",
    "decoder_final_ln": r"decoder.final_layer_norm",
}
all_model_shortcut_dict["gpt_neox"] = {
    "after_inner_dim": r".*[.]dense_4h_to_h",
    "before_inner_dim": r".*[.]query_key_value",
    "all_linears": r".*[.](dense_4h_to_h|dense_h_to_4h|query_key_value|dense)",
}
all_model_shortcut_dict["cpt"] = {
    "after_inner_dim": r".*[.]fc_out",
    "before_inner_dim": r".*[.](k_proj|rope_k_proj|v_proj)",
    "all_linears": r".*[.]fc_(in|out|linear)|.*[.](rope_q|rope_k|q|k|v|o)_proj",
}


@functools.lru_cache(maxsize=1)
def get_model_shortcut_dict(model_type):
    if model_type not in all_model_shortcut_dict:
        print(
            f"model shortcuts for the model {model_type} is not specified. Fallback to shared shortcuts."
        )
    return {
        **all_model_shortcut_dict[model_type],
        **all_model_shortcut_dict["shared"],
    }


def get_matched_pattern(model_shortcut_dict, key):
    if len(key) > 1 and key[0] == key[-1] and key[0] in ["'", '"']:
        return key.strip('"').strip("'")
    elif key in model_shortcut_dict:
        return model_shortcut_dict[key]
    else:
        print(
            f"Using {key} as a regular expression pattern since it is not in model_short_cut_dict."
        )
        return key


def get_model_re_pattern(model, model_shortcuts: Union[str, List[str]]):
    """Get the regular expression pattern for the model shortcuts.
    Args:
        model: a model object
        model_shortcuts: a string or a list of strings, each string is a model shortcut.
    Returns:
        model_re_pattern: a regular expression pattern for the model shortcuts.
    """
    if isinstance(model.torch_model, PreTrainedModel):
        model_type = model.torch_model.config.model_type
        model_shortcut_dict = get_model_shortcut_dict(model_type)
    else:
        raise NotImplementedError("Only support PreTrainedModel for now.")

    if isinstance(model_shortcuts, str):
        model_re_pattern = get_matched_pattern(model_shortcut_dict, model_shortcuts)
    else:
        model_re_pattern = "|".join(
            [get_matched_pattern(model_shortcut_dict, key) for key in model_shortcuts]
        )
    return model_re_pattern
