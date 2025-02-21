from typing import override

import torch
import torch.nn as nn
from util_obs_scaling_law_predictor import ObsScalingLawPredictor


class DirectDatePredictor(ObsScalingLawPredictor):
    """
    This class directly passes through Release Date
    """

    @override
    @staticmethod
    def fixed_benchmarks() -> list[str]:
        return ["release_date"]

    def __init__(
        self,
        benchmarks: list[str],
        benchmark_floors: list[float],
        train_model_scores: torch.Tensor,
    ):
        super().__init__(
            benchmarks,
            benchmark_floors,
            train_model_scores,
        )

        assert benchmarks == self.fixed_benchmarks()

        self.benchmarks = benchmarks
        self.intermediate = "release_date"
        self.benchmark_floors = benchmark_floors
        # we only store the logflop scores because they're the only thing we use
        # in M
        self.flop_scores = nn.Buffer(train_model_scores[:, 0])

        self.train_losses = []

    @override
    def predict_capability_scores_from_model_scores(
        self,
        model_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        The capability score is just the logflops score.
        """
        return model_scores[:, 0]

    @override
    def predict_benchmark_scores_from_capability_scores(
        self,
        capability_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        The only benchmark is the logflops, which is the capability score.
        """
        return capability_scores
