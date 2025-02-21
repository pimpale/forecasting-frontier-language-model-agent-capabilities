from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ObsScalingLawPredictor(nn.Module):
    """
    Parent class for all observational scaling law predictors.
    """

    benchmarks: list[str]
    intermediate: str

    def __init__(
        self,
        benchmarks: list[str],
        benchmark_floors: list[float],
        train_model_scores: torch.Tensor,
    ):
        super().__init__()
        pass

    @staticmethod
    def fixed_benchmarks() -> list[str] | None:
        """
        Return the list of benchmarks that are fixed for this predictor.
        These benchmarks must appear after the necessary benchmarks in the tensor that is passed into the constructor,
        and the order of the benchmarks must be the same as the order of the benchmarks in this list.
        """
        return None

    @staticmethod
    def necessary_benchmarks() -> list[str]:
        """
        Return the list of benchmarks that are necessary for this predictor.
        These benchmarks must appear first in the tensor that is passed into the constructor,
        and the order of the benchmarks must be the same as the order of the benchmarks in this list.
        Only one of necessary_benchmarks and fixed_benchmarks should be implemented.
        """
        return []

    def predict_capability_scores_from_model_scores(
        self,
        model_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict capability scores directly from model scores.
        """
        raise NotImplementedError

    def predict_benchmark_scores_from_capability_scores(
        self,
        capability_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict benchmark scores from capability scores.
        """
        raise NotImplementedError

    def fit(self):
        pass


PC1_EPS = 1e-4


class ScalingLaw(nn.Module):
    def __init__(
        self,
        benchmark: str,
        floor: float,
        # shape: (*, num_samples)
        capability_scores: torch.Tensor,
        # shape: (*, num_samples,)
        benchmark_scores: torch.Tensor,
        maybe_ceil: Optional[float] = None,
    ):
        super().__init__()

        # compute mean and std of capability scores so we can normalize scores later
        # shape: (*,)
        self.capability_scores_mean = nn.Buffer(
            torch.mean(capability_scores, dim=-1).detach()
        )
        # shape: (*,)
        self.capability_scores_std = nn.Buffer(
            torch.std(capability_scores, dim=-1).detach()
        )

        self.train_losses = []
        self.benchmark = benchmark
        self.benchmark_floor = nn.Buffer(torch.tensor(floor, dtype=torch.float32))

        if maybe_ceil is None:
            # shape: (*,)
            self.min_ceil = nn.Buffer(
                torch.clamp(torch.max(benchmark_scores, dim=-1).values, 0.8, 1)
            )
            self.max_ceil = nn.Buffer(torch.ones_like(self.min_ceil))
        else:
            self.min_ceil = nn.Buffer(
                maybe_ceil + torch.zeros_like(self.capability_scores_mean)
            )
            self.max_ceil = nn.Buffer(
                maybe_ceil + torch.zeros_like(self.capability_scores_mean)
            )

        self.capability_scores = nn.Buffer(capability_scores)
        self.benchmark_scores = nn.Buffer(benchmark_scores)
        self.benchmark_ceil_raw = nn.Parameter(
            torch.ones_like(self.capability_scores_mean)
        )
        self.alpha = nn.Parameter(torch.zeros_like(self.capability_scores_mean))
        self.beta = nn.Parameter(torch.ones_like(self.capability_scores_mean))

    @property
    def benchmark_ceil(self) -> torch.Tensor:
        return (self.max_ceil - self.min_ceil) * torch.sigmoid(
            self.benchmark_ceil_raw
        ) + self.min_ceil

    def predict_logit_scores(self, x: torch.Tensor) -> torch.Tensor:
        score_norm = (x - self.benchmark_floor) / (
            self.benchmark_ceil - self.benchmark_floor
        )
        score_norm_floored = torch.clamp(score_norm, PC1_EPS, 1 - PC1_EPS)
        return torch.log(score_norm_floored / (1 - score_norm_floored))

    def predict_benchmark_logit_scores(
        self,
        # shape: (*, num_samples)
        capability_scores: torch.Tensor,
    ) -> torch.Tensor:
        x = (
            capability_scores - self.capability_scores_mean.unsqueeze(-1)
        ) / self.capability_scores_std.unsqueeze(-1)
        return self.beta.unsqueeze(-1) * x + self.alpha.unsqueeze(-1)

    def infer_capability_score_from_benchmark_score(
        self, benchmark_score: torch.Tensor
    ) -> torch.Tensor:
        # account for the fact that the benchmark score is not necessarily in the range [0, 1]
        benchmark_score_scaled = (
            benchmark_score - self.benchmark_floor.unsqueeze(-1)
        ) / (self.benchmark_ceil - self.benchmark_floor).unsqueeze(-1)
        # first, compute logit of score
        logit_score = torch.log(benchmark_score_scaled / (1 - benchmark_score_scaled))
        # then, compute normalized capability score
        norm_capability_score = (
            logit_score - self.alpha.unsqueeze(-1)
        ) / self.beta.unsqueeze(-1)
        # then, compute capability score
        return self.capability_scores_std.unsqueeze(
            -1
        ) * norm_capability_score + self.capability_scores_mean.unsqueeze(-1)

    def forward(self, capability_scores: torch.Tensor) -> torch.Tensor:
        return (self.benchmark_ceil - self.benchmark_floor).unsqueeze(
            -1
        ) * torch.sigmoid(
            self.predict_benchmark_logit_scores(capability_scores)
        ) + self.benchmark_floor.unsqueeze(
            -1
        )

    # @torch.compile(fullgraph=True, dynamic=True)
    def train_loss(self):
        return F.mse_loss(self(self.capability_scores), self.benchmark_scores)

    def fit(
        self,
        # how many epochs to train for
        epochs: int = 500,
    ):
        """
        Fit the scaling law to the provided model and benchmark scores.
        """
        optimizer = optim.Adam(params=self.parameters(), lr=1e-1, fused=True)
        best_train_loss = float("inf")
        best_state_dict = self.state_dict()
        for i in range(epochs):
            optimizer.zero_grad()
            l = self.train_loss()
            if l < best_train_loss:
                best_train_loss = l
                best_state_dict = deepcopy(self.state_dict())
            l.backward()
            optimizer.step()
            self.train_losses.append(l.item())
        # load best state dict
        self.load_state_dict(best_state_dict)
        # get last loss
        self.train_losses.append(self.train_loss().item())
