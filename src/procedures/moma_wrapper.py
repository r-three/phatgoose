import gin

from src.procedures.procedure import Procedure


@gin.configurable(
    allowlist=[
        "model",
        "moma_calls",
    ]
)
class MoMaWrapper(Procedure):
    linking_fields = ["model"]

    def __init__(
        self,
        model,
        moma_calls=[],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.moma_calls = moma_calls

    def run(self):
        for moma_call in self.moma_calls:
            moma_call(self.model)

    def save_states(self):
        # TODO(Checkpointing): save results and rng state
        pass

    def recover_states(self):
        # TODO(Checkpointing): load results and rng state
        pass

    def get_description(self):
        return [
            f"Procedure class: {self.__class__.__name__}",
            f"Apply to {self.model.name} model {len(self.manipulations)} manipulations"
            f"{[manipulation.__name__ for manipulation in self.manipulations]}",
        ]
