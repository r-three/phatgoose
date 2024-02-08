import os

import gin


@gin.configurable(allowlist=["model_name_or_path", "model_class"])
def hf_torch_model(model_name_or_path, model_class=""):
    model_name_or_path = os.path.expandvars(model_name_or_path)
    model_class = os.path.expandvars(model_class)

    from transformers import (
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForMaskedLM,
        AutoModelForQuestionAnswering,
        AutoModelForSeq2SeqLM,
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
    )

    model_class = {
        "": AutoModel,
        "causal_lm": AutoModelForCausalLM,
        "masked_lm": AutoModelForMaskedLM,
        "seq2seq_lm": AutoModelForSeq2SeqLM,
        "seq_cls": AutoModelForSequenceClassification,
        "token_cls": AutoModelForTokenClassification,
        "qa": AutoModelForQuestionAnswering,
    }[model_class]
    torch_model = model_class.from_pretrained(model_name_or_path)

    return torch_model


@gin.configurable(allowlist=["model_name_or_path"])
def hf_tokenizer(model_name_or_path):
    model_name_or_path = os.path.expandvars(model_name_or_path)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if model_name_or_path.startswith("EleutherAI/pythia"):
        tokenizer.add_special_tokens({"pad_token": "<|padding|>"})
    assert tokenizer.pad_token_id is not None

    test_tokens = tokenizer.build_inputs_with_special_tokens([-100])
    if test_tokens[0] != -100:
        tokenizer.bos_token_id = test_tokens[0]
    else:
        tokenizer.bos_token_id = None
    if test_tokens[-1] != -100:
        tokenizer.eos_token_id = test_tokens[-1]
    else:
        tokenizer.eos_token_id = None

    return tokenizer
