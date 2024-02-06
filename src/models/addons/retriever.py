import gin
import torch
from sentence_transformers import SentenceTransformer

from src.models.addons.addon import Addon


@gin.configurable
class FeatureExtractor(Addon):
    has_pre_forward = True

    def __init__(
        self,
        host_module,
        global_hidden_dict,
        write_hidden_key,
        model_name="all-MiniLM-L6-v2",
        include_answer_choices=False,
    ):
        super().__init__(global_hidden_dict)
        self.write_hidden_key = write_hidden_key
        self.model_name = model_name
        self.sentence_transformer = SentenceTransformer(model_name)
        self.include_answer_choices = include_answer_choices

    def pre_forward(self, *args, **kwargs):
        all_embeddings = []
        input_strings = self.global_hidden_dict["batch_input"]["input_str"]
        if "answer_choices" in self.global_hidden_dict["batch_input"]:
            answer_choice_strings = self.global_hidden_dict["batch_input"][
                "answer_choices"
            ]
        else:
            answer_choice_strings = None
        for i in range(len(input_strings)):
            if self.include_answer_choices:
                if answer_choice_strings:
                    answer_choice_str = ", ".join(answer_choice_strings[i])
                else:
                    answer_choice_str = "None"
                text = (
                    f"Answer Choices: {answer_choice_str}, Instance: {input_strings[i]}"
                )
            else:
                text = input_strings[i]
            embedding = torch.tensor(self.sentence_transformer.encode(text)).to(
                kwargs["input_ids"].device
            )
            all_embeddings.append(embedding)
        input_features = torch.stack(all_embeddings)
        num_choices = kwargs["input_ids"].shape[0] / input_features.shape[0]
        assert num_choices.is_integer(), "Number of choices must be an integer"
        num_choices = int(num_choices)
        input_features = input_features.repeat_interleave(num_choices, dim=0)
        self.global_hidden_dict[self.write_hidden_key] = input_features
