import json
import os
import pickle
from collections import defaultdict

import gin
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import entropy


class AnalysisProcessor(object):
    def __init__(self):
        pass

    def batch_process(self, batch_inputs, batch_outputs, global_hidden_dict):
        raise NotImplementedError()

    def dataset_process(self, dataset_name):
        raise NotImplementedError()

    def cross_dataset_process(self):
        raise NotImplementedError()


@gin.configurable
class RoutingDistribution(AnalysisProcessor):
    def __init__(self, save_dir):
        self.name = "routing_distribution"
        self.routing_dist_per_dataset = {}
        self.routing_dist = defaultdict(dict)
        self.count_per_dataset = {}
        self.save_dir = os.path.expandvars(save_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def batch_process(self, batch_inputs, batch_outputs, global_hidden_dict):
        for key in global_hidden_dict:
            if key[0] == "routing_weights":
                routing_weights = global_hidden_dict[key].float().cpu()
                if len(routing_weights.shape) == 2:
                    # sentence level routing
                    routing_weights = routing_weights.numpy()
                    if key in self.routing_dist_per_dataset:
                        self.routing_dist_per_dataset[key] += np.sum(
                            routing_weights, axis=0
                        )
                        self.count_per_dataset[key] += routing_weights.shape[0]
                    else:
                        self.routing_dist_per_dataset[key] = np.sum(
                            routing_weights, axis=0
                        )
                        self.count_per_dataset[key] = routing_weights.shape[0]
                elif len(routing_weights.shape) == 3:
                    # token level routing
                    if (
                        "encoder" in key[2]
                        or "EncDecAttention.k" in key[2]
                        or "EncDecAttention.v" in key[2]
                    ):
                        # encoder tokens routing weights
                        encoder_mask = (
                            global_hidden_dict[("mask", "prepare_mask", "encoder")]
                            .float()
                            .cpu()
                        )
                        num_experts = routing_weights.shape[-1]
                        token_routing_weights = routing_weights.reshape(-1, num_experts)
                        mask = encoder_mask.reshape(-1)
                    elif "decoder" in key[2]:
                        # decoder tokens routing weights
                        if "prediction" in batch_outputs:
                            # classification
                            answer_choices_ids = batch_inputs["answer_choices_ids"]
                            (
                                batch_size,
                                num_choices,
                                choice_length,
                            ) = answer_choices_ids.shape
                            prediction = batch_outputs["prediction"].cpu()
                            routing_weights = routing_weights.reshape(
                                batch_size, num_choices, choice_length, -1
                            )
                            num_experts = routing_weights.shape[-1]
                            predicted_choice_routing_weights = torch.gather(
                                routing_weights,
                                1,
                                prediction[:, None, None, None].repeat(
                                    1, 1, choice_length, num_experts
                                ),
                            ).squeeze(1)
                            predicted_choice_ids = torch.gather(
                                answer_choices_ids,
                                1,
                                prediction[:, None, None].repeat(1, 1, choice_length),
                            ).squeeze(1)
                            token_routing_weights = (
                                predicted_choice_routing_weights.reshape(
                                    -1, num_experts
                                )
                            )
                            mask = predicted_choice_ids.reshape(-1)
                        elif "output_ids" in batch_outputs:
                            if key[-1] != "concatenated":
                                continue
                            output_ids = batch_outputs["output_ids"].cpu()[:, 1:]
                            num_experts = routing_weights.shape[-1]
                            token_routing_weights = routing_weights.reshape(
                                -1, num_experts
                            )
                            mask = output_ids.reshape(-1)
                    else:
                        raise Exception("Unknown routing weights key")
                    token_routing_weights = token_routing_weights[mask != 0]
                    if key[2] in self.routing_dist_per_dataset:
                        self.routing_dist_per_dataset[key[2]] += torch.sum(
                            token_routing_weights, dim=0
                        )
                        self.count_per_dataset[key[2]] += token_routing_weights.shape[0]
                    else:
                        self.routing_dist_per_dataset[key[2]] = torch.sum(
                            token_routing_weights, dim=0
                        )
                        self.count_per_dataset[key[2]] = token_routing_weights.shape[0]

    def dataset_process(self, dataset_name):
        for key in self.routing_dist_per_dataset:
            self.routing_dist[dataset_name][key] = (
                self.routing_dist_per_dataset[key] / self.count_per_dataset[key]
            )
            self.routing_dist[dataset_name][key] = self.routing_dist[dataset_name][
                key
            ].numpy()
        self.routing_dist_per_dataset = {}
        self.count_per_dataset = {}

    def cross_dataset_process(self):
        pass

    def save(self, step=None):
        for dataset_name in self.routing_dist:
            save_path = self.save_dir + f"/{dataset_name.replace('/', '_')}.pickle"
            with open(save_path, "wb") as f:
                pickle.dump(self.routing_dist[dataset_name], f)


@gin.configurable
class EntropyDistribution(AnalysisProcessor):
    def __init__(self, save_dir):
        self.name = "entropy_distribution"
        self.entropy_values_per_dataset = {}
        self.entropy_values = defaultdict(dict)
        self.save_dir = os.path.expandvars(save_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def batch_process(self, batch_inputs, batch_outputs, global_hidden_dict):
        for key in global_hidden_dict:
            if key[0] == "routing_weights":
                routing_weights = global_hidden_dict[key].float().cpu().numpy()
                entropy_values = entropy(routing_weights.T, base=2)
                if key[2] not in self.entropy_values_per_dataset:
                    self.entropy_values_per_dataset[key[2]] = []
                self.entropy_values_per_dataset[key[2]].extend(entropy_values)

    def dataset_process(self, dataset_name):
        for key in self.entropy_values_per_dataset:
            self.entropy_values[dataset_name][key] = self.entropy_values_per_dataset[
                key
            ]
        self.entropy_values_per_dataset = {}

    def cross_dataset_process(self):
        pass

    def save(self, step=None):
        # plot a histogram wtih entropy values
        for dataset_name in self.entropy_values:
            dataset_save_dir = self.save_dir + f"/{dataset_name.replace('/', '_')}"
            if not os.path.exists(dataset_save_dir):
                os.makedirs(dataset_save_dir)
            for key in self.entropy_values[dataset_name]:
                save_path = dataset_save_dir + f"/{key}.png"
                x = self.entropy_values[dataset_name][key]
                q25, q75 = np.percentile(x, [25, 75])
                bin_width = 2 * (q75 - q25) * len(x) ** (-1 / 3)
                if bin_width < 1:
                    bins = 10
                else:
                    bins = round((max(x) - min(x)) / bin_width)
                plt.hist(x, bins=bins)
                plt.savefig(save_path)
                plt.close()


@gin.configurable
class SaveAveragedHiddens(AnalysisProcessor):
    def __init__(self, save_dir):
        self.name = "averaged_hiddens"
        self.averaged_hiddens_per_dataset = {}
        self.averaged_hiddens = defaultdict(dict)
        self.count_per_dataset = {}
        self.save_dir = os.path.expandvars(save_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def batch_process(self, batch_inputs, batch_outputs, global_hidden_dict):
        for key in global_hidden_dict:
            if key[0] == "hidden_states":
                # hidden_states = global_hidden_dict[key].float().cpu()
                # layer_norm = torch.nn.LayerNorm(
                #     hidden_states.shape[-1], elementwise_affine=False
                # )
                # hidden_states = layer_norm(hidden_states)
                # hidden_states = hidden_states.numpy()
                hidden_states = global_hidden_dict[key].float().cpu().numpy()
                if key[2] in self.averaged_hiddens_per_dataset:
                    self.averaged_hiddens_per_dataset[key[2]] += np.sum(
                        hidden_states, axis=0
                    )
                    self.count_per_dataset[key[2]] += hidden_states.shape[0]
                else:
                    self.averaged_hiddens_per_dataset[key[2]] = np.sum(
                        hidden_states, axis=0
                    )
                    self.count_per_dataset[key[2]] = hidden_states.shape[0]

    def dataset_process(self, dataset_name):
        for key in self.averaged_hiddens_per_dataset:
            self.averaged_hiddens[dataset_name][key] = (
                self.averaged_hiddens_per_dataset[key] / self.count_per_dataset[key]
            )
        self.averaged_hiddens_per_dataset = {}
        self.count_per_dataset = {}

    def cross_dataset_process(self):
        pass

    def save(self, step=None):
        for dataset_name in self.averaged_hiddens:
            save_path = self.save_dir + f"/{dataset_name.replace('/', '_')}.pickle"
            with open(save_path, "wb") as f:
                pickle.dump(self.averaged_hiddens[dataset_name], f)


@gin.configurable
class WriteOutputText(AnalysisProcessor):
    def __init__(self, save_dir):
        self.name = "generated_output"
        self.generated_output_per_dataset = []
        self.generated_output = defaultdict(dict)
        self.save_dir = os.path.expandvars(save_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def batch_process(self, batch_inputs, batch_outputs, global_hidden_dict):
        if "output_text" in batch_outputs.keys():
            # generation
            input_strs = batch_inputs["input_str"]
            output_text = batch_outputs["output_text"]
            target_strs = batch_inputs["target_str"]
            references = batch_inputs["references"]
            for i in range(len(input_strs)):
                self.generated_output_per_dataset.append(
                    {
                        "input_str": input_strs[i],
                        "output_text": output_text[i],
                        "target_str": target_strs[i],
                        "references": references[i],
                    }
                )
        else:
            # classification
            input_strs = batch_inputs["input_str"]
            answer_choices = batch_inputs["answer_choices"]
            label = batch_inputs["label"]
            prediction = batch_outputs["prediction"]
            if "multi_label" in batch_inputs:
                multi_label = batch_inputs["multi_label"]
            else:
                multi_label = None
            for i in range(len(input_strs)):
                self.generated_output_per_dataset.append(
                    {
                        "input_str": input_strs[i],
                        "answer_choices": answer_choices[i],
                        "label": label[i].item(),
                        "prediction": prediction[i].item(),
                        "multi_label": multi_label[i]
                        if multi_label is not None
                        else None,
                    }
                )

    def dataset_process(self, dataset_name):
        self.generated_output[dataset_name] = self.generated_output_per_dataset
        self.generated_output_per_dataset = []

    def cross_dataset_process(self):
        pass

    def save(self, step=None):
        for dataset_name in self.generated_output:
            save_path = self.save_dir + f"/{dataset_name.replace('/', '_')}.json"
            with open(save_path, "w") as f:
                json.dump(self.generated_output[dataset_name], f, indent=4)
