import inspect
from collections import OrderedDict

import gin
import torch
import torch.nn.functional as F

from src.utils.constants import FLOAT_EPSILON


def get_attention_mask(token_ids, tokenizer):
    return (token_ids != tokenizer.pad_token_id).long()


def replace_eos_by_pad(token_ids, tokenizer):
    return token_ids.masked_fill(
        token_ids == tokenizer.eos_token_id, tokenizer.pad_token_id
    )


def align(token_ids, tokenizer, right_align=False):
    non_pad_token_mask = token_ids != tokenizer.pad_token_id
    non_pad_length = non_pad_token_mask.sum(dim=-1, keepdim=True)
    arange_tensor = torch.ones_like(token_ids).cumsum(dim=-1)
    new_non_pad_token_mask = arange_tensor <= non_pad_length
    if right_align:
        new_non_pad_token_mask = new_non_pad_token_mask.flip([-1])
    new_token_ids = token_ids.new_ones(token_ids.size()) * tokenizer.pad_token_id
    new_token_ids[new_non_pad_token_mask] = token_ids[non_pad_token_mask]
    return new_token_ids


def shift_pad(token_ids, tokenizer):
    non_pad_token_mask = token_ids != tokenizer.pad_token_id
    new_non_pad_token_mask = non_pad_token_mask.flip([-1])
    new_token_ids = torch.ones_like(token_ids) * tokenizer.pad_token_id
    new_token_ids[new_non_pad_token_mask] = token_ids[non_pad_token_mask]
    return new_token_ids


def prepare_label(token_ids, tokenizer, shift=False):
    label_ids = token_ids.masked_fill(token_ids == tokenizer.pad_token_id, -100)
    if shift:
        head, remain = label_ids.split([1, label_ids.size(-1) - 1], dim=-1)
        label_ids = torch.cat([remain, head.fill_(-100)], dim=-1)
    return label_ids


def pytree_expand(pytree, num_replicas, dim=0, flatten=False):
    if pytree is None:
        return None
    elif isinstance(pytree, torch.Tensor):
        sizes = [-1] * pytree.ndim
        sizes[dim] = num_replicas
        if flatten:
            return pytree.repeat_interleave(num_replicas, dim)
        else:
            return pytree.expand(*sizes)
    elif isinstance(pytree, (list, tuple)):
        return type(pytree)(
            pytree_expand(x, num_replicas, dim, flatten=flatten) for x in pytree
        )
    elif isinstance(pytree, dict):
        return type(pytree)(
            (k, pytree_expand(v, num_replicas, dim, flatten=flatten))
            for k, v in pytree.items()
        )
    else:
        raise ValueError(f"Unsupported type {type(pytree)}")


def prepare_beginning(input_ids, tokenizer):
    input_mask = get_attention_mask(input_ids, tokenizer)
    input_labels = prepare_label(input_ids, tokenizer, shift=True).masked_fill(
        input_mask == 0, -100
    )
    if tokenizer.bos_token_id is None:
        input_ids, tail = input_ids.split([input_ids.size(-1) - 1, 1], dim=-1)
        input_mask = input_mask[..., 1:]
        input_labels = input_labels[..., 1:]
    else:
        tail = None
    return input_ids, input_mask, input_labels, tail


def prepare_continuation(answer_choices_ids, tokenizer, input_tail=None):
    if tokenizer.bos_token_id is None:
        answer_choices_ids = torch.cat([input_tail, answer_choices_ids], dim=-1)
    answer_choices_labels = prepare_label(answer_choices_ids, tokenizer, shift=True)
    answer_choices_mask = get_attention_mask(answer_choices_ids, tokenizer)
    return answer_choices_ids, answer_choices_mask, answer_choices_labels


def multiple_choice_by_perplexity(
    output_logits,
    choices_labels,
    labels,
    length_normalization=True,
    multiple_choice_loss=1.0,
    unlikelihood_loss=1.0,
):
    batch_size, num_choices, _, vocab_size = output_logits.size()
    token_loss = F.cross_entropy(
        output_logits.flatten(end_dim=-2),
        choices_labels.flatten(),
        reduction="none",
    ).view(batch_size, num_choices, -1)

    choices_scores = token_loss.sum(dim=-1)
    if length_normalization:
        choices_scores = choices_scores / ((choices_labels >= 0).sum(dim=-1) + 1e-6)
    # some choices can be just padding to account for different number of answer choices for different examples
    mask = torch.all(choices_labels == -100, dim=-1)
    choices_scores[mask] = float("inf")
    lm_loss = F.cross_entropy(
        output_logits[range(batch_size), labels].flatten(end_dim=-2),
        choices_labels[range(batch_size), labels].flatten(),
    )

    prediction = choices_scores.argmin(dim=-1)
    if multiple_choice_loss > 0:
        mc_loss = F.cross_entropy(-choices_scores, labels)
    else:
        mc_loss = 0

    if unlikelihood_loss > 0:
        # p(other) = 1 - p(label) = 1 - exp(-token_loss)
        distraction_loglikely = (-token_loss).masked_fill(
            choices_labels < 0, float("-inf")
        )  # ignore padding
        distraction_loglikely[range(batch_size), labels] = float(
            "-inf"
        )  # exclude the correct choice
        ul_loss = (
            -torch.log(1 - torch.exp(distraction_loglikely) + FLOAT_EPSILON).sum()
            / (distraction_loglikely != float("-inf")).sum()
        )
    else:
        ul_loss = 0

    return (
        lm_loss,
        mc_loss * multiple_choice_loss,
        ul_loss * unlikelihood_loss,
        prediction,
    )


@gin.configurable
class InterfaceMixin:
    def __init__(
        self,
        language_modeling_interface=None,
        generation_interface=None,
        mutiple_choice_interface=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.interface_dict = OrderedDict()
        self.accepted_kwargs_dict = OrderedDict()

        if language_modeling_interface is not None:
            self.interface_dict["lm"] = {
                "lm_4encdec": InterfaceMixin.language_modeling_for_encdec,
                "lm_4dec": InterfaceMixin.language_modeling_for_decoder,
            }[language_modeling_interface]
        if generation_interface is not None:
            self.interface_dict["gen"] = {
                "gen_4encdec": InterfaceMixin.generation_for_encdec,
                "gen_4dec": InterfaceMixin.generation_for_decoder,
            }[generation_interface]
        if mutiple_choice_interface is not None:
            self.interface_dict["mc"] = {
                "mc_byppl_4encdec": InterfaceMixin.multiple_choice_by_perplexity_for_encdec,
                "mc_byppl_4dec": InterfaceMixin.multiple_choice_by_perplexity_for_decoder,
                "mc_byppl_4encdec_fast": InterfaceMixin.multiple_choice_by_perplexity_for_encdec_fast,
                "mc_byppl_4dec_fast": InterfaceMixin.multiple_choice_by_perplexity_for_decoder_fast,
            }[mutiple_choice_interface]

        for interface_name, interface_func in self.interface_dict.items():
            self.accepted_kwargs_dict[interface_name] = set(
                inspect.signature(interface_func).parameters
            )
            for excluded_kwarg in ["torch_model", "tokenizer", "batch_input"]:
                self.accepted_kwargs_dict[interface_name].remove(excluded_kwarg)

    @staticmethod
    def language_modeling_for_encdec(torch_model, tokenizer, batch_input):
        global_hidden_updates = {}
        input_ids = batch_input["input_ids"]
        target_ids = batch_input["target_ids"]
        attention_mask = get_attention_mask(input_ids, tokenizer)

        target_labels = prepare_label(target_ids, tokenizer, shift=False)
        model_output = torch_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=target_labels,
        )
        batch_output = {
            "loss": 0.0,
        }
        global_hidden_updates[("loss", "interface", "lm")] = model_output.loss
        return batch_output, global_hidden_updates

    @staticmethod
    def language_modeling_for_decoder(
        torch_model,
        tokenizer,
        batch_input,
        input_loss=1.0,
    ):
        global_hidden_updates = {}
        input_ids = shift_pad(batch_input["input_ids"], tokenizer)
        target_ids = batch_input["target_ids"]

        input_attention_mask = get_attention_mask(input_ids, tokenizer)
        input_labels = prepare_label(input_ids, tokenizer)
        model_output_part1 = torch_model(
            input_ids=input_ids,
            attention_mask=input_attention_mask,
            labels=input_labels,
        )
        full_attention_mask = torch.cat(
            [input_attention_mask, get_attention_mask(target_ids, tokenizer)], dim=-1
        )
        target_labels = prepare_label(target_ids, tokenizer)
        model_output_part2 = torch_model(
            input_ids=target_ids,
            attention_mask=full_attention_mask,
            labels=target_labels,
            past_key_values=model_output_part1.past_key_values,
        )
        batch_output = {
            "loss": 0.0,
            "logits": model_output_part2.logits,
        }
        global_hidden_updates[("loss", "interface", "lm")] = model_output_part2.loss
        if input_loss > 0:
            global_hidden_updates[("loss", "interface", "input_lm")] = (
                model_output_part1.loss * input_loss
            )
        return batch_output, global_hidden_updates

    @staticmethod
    def generation_for_encdec(
        torch_model, tokenizer, batch_input, num_beams=1, max_gen_length=20
    ):
        global_hidden_updates = {}
        input_ids = batch_input["input_ids"]

        attention_mask = get_attention_mask(input_ids, tokenizer)
        output_ids = torch_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=num_beams,
            max_length=max_gen_length,
        )
        output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        batch_output = {"output_ids": output_ids, "output_text": output_text}
        return batch_output, global_hidden_updates

    @staticmethod
    def generation_for_decoder(
        torch_model, tokenizer, batch_input, num_beams=1, max_gen_length=20
    ):
        global_hidden_updates = {}
        input_ids = shift_pad(batch_input["input_ids"], tokenizer)
        input_ids = torch.cat(
            [
                input_ids,
                torch.ones_like(input_ids[..., 0:1]) * tokenizer.bos_token_id,
            ],
            dim=-1,
        )
        input_length = input_ids.size(-1)

        attention_mask = get_attention_mask(input_ids, tokenizer)
        output_ids = torch_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=num_beams,
            max_length=max_gen_length + input_length,
        )
        output_ids = output_ids[..., input_length:]
        output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        batch_output = {"output_ids": output_ids, "output_text": output_text}
        return batch_output, global_hidden_updates

    @staticmethod
    def multiple_choice_by_perplexity_for_encdec(
        torch_model,
        tokenizer,
        batch_input,
        length_normalization=True,
        multiple_choice_loss=1.0,
        unlikelihood_loss=1.0,
    ):
        global_hidden_updates = {}
        batch_size, num_choices = batch_input["answer_choices_ids"].size()[:2]
        choices_ids = batch_input["answer_choices_ids"].flatten(0, 1)
        input_ids = batch_input["input_ids"].repeat_interleave(num_choices, dim=0)
        labels = batch_input["label"]

        attention_mask = get_attention_mask(input_ids, tokenizer)
        choices_labels = prepare_label(choices_ids, tokenizer, shift=False)
        model_output = torch_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=choices_labels,
        )
        lm_loss, mc_loss, ul_loss, prediction = multiple_choice_by_perplexity(
            model_output.logits.view(
                batch_size, num_choices, *model_output.logits.size()[1:]
            ),
            choices_labels.view(batch_size, num_choices, *choices_labels.size()[1:]),
            labels,
            length_normalization,
            multiple_choice_loss,
            unlikelihood_loss,
        )

        global_hidden_updates[("loss", "interface", "lm")] = lm_loss
        if multiple_choice_loss > 0.0:
            global_hidden_updates[("loss", "interface", "mc")] = mc_loss
        if unlikelihood_loss > 0.0:
            global_hidden_updates[("loss", "interface", "ul")] = ul_loss

        batch_output = {
            "loss": 0.0,
            "prediction": prediction,
        }

        return batch_output, global_hidden_updates

    @staticmethod
    def multiple_choice_by_perplexity_for_encdec_fast(
        torch_model,
        tokenizer,
        batch_input,
        length_normalization=True,
        multiple_choice_loss=1.0,
        unlikelihood_loss=1.0,
    ):
        global_hidden_updates = {}
        choices_ids = batch_input["answer_choices_ids"]
        input_ids = batch_input["input_ids"].unsqueeze(1)
        labels = batch_input["label"]

        attention_mask = get_attention_mask(input_ids, tokenizer)
        decoder_attention_mask = get_attention_mask(choices_ids)
        choices_labels = prepare_label(choices_ids, tokenizer, shift=True)
        model_output = torch_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=choices_ids,
            decoder_attention_mask=decoder_attention_mask,
        )
        loss, prediction = multiple_choice_by_perplexity(
            model_output.logits,
            choices_labels,
            labels,
            length_normalization,
            multiple_choice_loss,
            unlikelihood_loss,
        )

        batch_output = {
            "loss": 0.0,
            "prediction": prediction,
        }
        global_hidden_updates = {
            ("loss", "interface", "lm"): loss,
        }

        return batch_output, global_hidden_updates

    @staticmethod
    def _multiple_choice_by_perplexity_for_decoder(
        torch_model,
        tokenizer,
        batch_input,
        length_normalization,
        multiple_choice_loss,
        unlikelihood_loss,
        input_loss,
        expand,
    ):
        """
        If expand is False, the model will be more memory efficient. But it depends on special implementation inside the model.
        The model will need to handle past_key_query that doesn't have the same size as the input.
        """
        global_hidden_updates = {}
        batch_size, num_choices = batch_input["answer_choices_ids"].size()[:2]

        input_ids, input_mask, input_labels, input_tail = prepare_beginning(
            shift_pad(batch_input["input_ids"], tokenizer), tokenizer
        )
        if input_tail is not None:
            input_tail = input_tail.repeat_interleave(num_choices, dim=0)
        (
            answer_choices_ids,
            answer_choices_mask,
            answer_choices_labels,
        ) = prepare_continuation(
            batch_input["answer_choices_ids"].flatten(0, 1), tokenizer, input_tail
        )
        full_attention_mask = torch.cat(
            [input_mask.repeat_interleave(num_choices, dim=0), answer_choices_mask],
            dim=1,
        )

        model_output_part1 = torch_model(
            input_ids=input_ids,
            attention_mask=input_mask,
            use_cache=True,
        )
        if expand:
            past_key_values = pytree_expand(
                model_output_part1.past_key_values, num_choices, dim=0, flatten=True
            )
        else:
            past_key_values = model_output_part1.past_key_values

        model_output_part2 = torch_model(
            input_ids=answer_choices_ids,
            attention_mask=full_attention_mask,
            past_key_values=past_key_values,
            use_cache=False,
        )

        if input_loss > 0.0:
            input_loss = (
                F.cross_entropy(
                    model_output_part1.logits.flatten(end_dim=-2),
                    input_labels.flatten(),
                    reduction="mean",
                )
                * input_loss
            )

            global_hidden_updates[("loss", "interface", "input_lm")] = input_loss
        lm_loss, mc_loss, ul_loss, prediction = multiple_choice_by_perplexity(
            model_output_part2.logits.view(
                batch_size, num_choices, *model_output_part2.logits.size()[1:]
            ),
            answer_choices_labels.view(
                batch_size, num_choices, *answer_choices_labels.size()[1:]
            ),
            batch_input["label"],
            length_normalization,
            multiple_choice_loss,
            unlikelihood_loss,
        )
        global_hidden_updates[("loss", "interface", "lm")] = lm_loss
        if multiple_choice_loss > 0.0:
            global_hidden_updates[("loss", "interface", "mc")] = mc_loss
        if unlikelihood_loss > 0.0:
            global_hidden_updates[("loss", "interface", "ul")] = ul_loss

        batch_output = {
            "loss": 0.0,
            "prediction": prediction,
        }
        return batch_output, global_hidden_updates

    @staticmethod
    def multiple_choice_by_perplexity_for_decoder(
        torch_model,
        tokenizer,
        batch_input,
        length_normalization=True,
        multiple_choice_loss=0.0,
        unlikelihood_loss=0.0,
        input_loss=0.0,
    ):
        return InterfaceMixin._multiple_choice_by_perplexity_for_decoder(
            torch_model,
            tokenizer,
            batch_input,
            length_normalization,
            multiple_choice_loss,
            unlikelihood_loss,
            input_loss,
            expand=True,
        )

    @staticmethod
    def multiple_choice_by_perplexity_for_decoder_fast(
        torch_model,
        tokenizer,
        batch_input,
        length_normalization=True,
        multiple_choice_loss=0.0,
        unlikelihood_loss=0.0,
        input_loss=0.0,
    ):
        return InterfaceMixin._multiple_choice_by_perplexity_for_decoder(
            torch_model,
            tokenizer,
            batch_input,
            length_normalization,
            multiple_choice_loss,
            unlikelihood_loss,
            input_loss,
            expand=False,
        )

    def __call__(self, batch_input, interface_info):
        """
        Find an interface_func speicified by interface_info (data-dependent) and interface_dict (model-dependent)
        Call the interface_func with batch_input and additional kwargs from interface_info.
        """
        batch_interface = interface_info.interface
        interface_func = self.interface_dict[batch_interface]
        interface_kwargs = {
            key: getattr(interface_info, key)
            for key in self.accepted_kwargs_dict[batch_interface]
            if hasattr(interface_info, key)
        }

        batch_output, global_hidden_updates = interface_func(
            self.torch_model, self.tokenizer, batch_input, **interface_kwargs
        )
        self.global_hidden_dict.update(global_hidden_updates)

        if "loss" in batch_output:
            for key, value in self.global_hidden_dict.items():
                if isinstance(key, tuple) and key[0] == "loss":
                    if isinstance(value, list):
                        self.global_hidden_dict[key] = torch.stack(value).mean()
                        batch_output["loss"] += self.global_hidden_dict[key]
                    else:
                        batch_output["loss"] += value

        return batch_output
