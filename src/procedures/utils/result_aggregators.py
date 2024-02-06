# TODO: This feels too loose. It's decomposed into so many parts that it become hard to understand.
# As if we implement a neural network only from matmul without any abstraction.
import numpy as np
from scipy.stats import iqr


class Aggregator(object):
    def __init__(self):
        ...

    def __call__(self, results):
        raise NotImplementedError()

    @staticmethod
    def _reduction(values, reduction):
        if reduction == "mean":
            return (np.mean(values), np.std(values))
        elif reduction == "median":
            return (np.median(values), iqr(values))
        else:
            raise ValueError(f"Reduction {reduction} not supported.")


class MainAggregator(Aggregator):
    def __init__(
        self,
        reduction="mean",
        convert_fn=lambda x: x["score"],
    ):
        self.reduction = reduction
        self.convert_fn = convert_fn

    def __call__(self, results):
        main_metric_values = [
            self.convert_fn(dataset_results) for dataset_results in results.values()
        ]
        average, confidence = self._reduction(main_metric_values, self.reduction)
        metrics_to_add = {
            "average_score": average,
            "confidence": confidence,
        }

        results.update(metrics_to_add)


class MultiMetricAggregator(Aggregator):
    def __init__(
        self, reduction="mean", metric_names=["accuracy", "f1", "exact_match"]
    ):
        self.reduction = reduction
        self.metric_names = metric_names

    def __call__(self, results):
        metrics_to_add = {}
        for metric_name in self.metric_names:
            # TODO: Is this what is intended?
            metric_values = [
                dataset_results[metric_name]
                for dataset_results in results.values()
                if metric_name in dataset_results
            ]
            metrics_to_add[metric_name] = self._reduction(metric_values, self.reduction)

        results.update(metrics_to_add)


class DatasetGroupAggregator(Aggregator):
    def __init__(self):
        raise NotImplementedError()
