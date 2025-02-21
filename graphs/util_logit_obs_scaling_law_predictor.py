from typing import Optional, override

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from util_obs_scaling_law_predictor import ObsScalingLawPredictor

PC1_EPS = 1e-4


class LogitPC1Predictor(ObsScalingLawPredictor):
    def __init__(
        self,
        benchmarks: list[str],
        benchmark_floors: list[float],
        train_model_scores: torch.Tensor,
        benchmark_ceilings: Optional[list[float]] = None,
    ):
        super().__init__(
            benchmarks,
            benchmark_floors,
            train_model_scores,
        )

        self.train_losses = []
        self.benchmark_ceilings = benchmark_ceilings

        B = len(benchmarks)
        self.benchmarks = benchmarks
        self.intermediate = "PC-1"

        # in B
        self.benchmark_floors = nn.Buffer(
            torch.tensor(benchmark_floors, dtype=torch.float32)
        )

        # note: We initialize with values that are likely to be close to the true values, but the model will learn them in training

        self.benchmark_ceil_raw = nn.Parameter(
            torch.full((B,), fill_value=-4, dtype=torch.float32)
        )

        # in M x B
        self.train_model_scores = nn.Buffer(train_model_scores)

        # in B
        self.benchmark_weights = nn.Parameter(torch.ones(B, dtype=torch.float32))
        self.alpha = nn.Parameter(torch.zeros(B, dtype=torch.float32))
        self.beta = nn.Parameter(torch.full((B,), fill_value=0.5, dtype=torch.float32))

    @property
    def benchmark_ceil(self) -> torch.Tensor:
        if self.benchmark_ceilings is not None:
            return torch.tensor(self.benchmark_ceilings, dtype=torch.float32)
        else:
            min_ceil = torch.clamp(
                torch.max(self.train_model_scores, dim=0).values, 0.8, 1
            )
            return (1 - min_ceil) * torch.sigmoid(self.benchmark_ceil_raw) + min_ceil

    def predict_logit_scores(self, model_scores: torch.Tensor) -> torch.Tensor:
        score_norm = (model_scores - self.benchmark_floors) / (
            self.benchmark_ceil - self.benchmark_floors
        )
        score_norm_floored = torch.clamp(score_norm, PC1_EPS, 1 - PC1_EPS)
        return torch.log(score_norm_floored / (1 - score_norm_floored))

    @property
    def pca_benchmark_weights(self) -> torch.Tensor:
        # perform PCA and get the first component, return it
        # logit_scores: M x B
        # benchmark_weights: B x 1
        U, S, V = torch.pca_lowrank(self.predict_logit_scores(self.train_model_scores))
        # gamma in B x 1
        return -V[:, 0]

    def predict_capability_scores(self, logit_scores: torch.Tensor) -> torch.Tensor:
        # logit_scores: M x B
        # benchmark_weights: B x 1
        benchmark_weights = self.benchmark_weights.unsqueeze(1)
        # benchmark_weights = self.pca_benchmark_weights.unsqueeze(1)
        # M x 1
        capability_score = logit_scores @ benchmark_weights
        return capability_score.squeeze(1)

    @override
    def predict_capability_scores_from_model_scores(
        self, model_scores: torch.Tensor
    ) -> torch.Tensor:
        return self.predict_capability_scores(self.predict_logit_scores(model_scores))

    def predict_benchmark_logit_scores(
        self, capability_scores: torch.Tensor
    ) -> torch.Tensor:
        # capability_scores: M x 1
        capability_scores = capability_scores.unsqueeze(1)
        # beta = 1 x B
        beta = self.beta.unsqueeze(0)
        # M x B
        predicted_logit_scores = capability_scores @ beta + self.alpha
        return predicted_logit_scores

    def predict_benchmark_scores(
        self, benchmark_logit_scores: torch.Tensor
    ) -> torch.Tensor:
        # in M x B
        return (self.benchmark_ceil - self.benchmark_floors) * torch.sigmoid(
            benchmark_logit_scores
        ) + self.benchmark_floors

    @override
    def predict_benchmark_scores_from_capability_scores(
        self, capability_scores: torch.Tensor
    ) -> torch.Tensor:
        return self.predict_benchmark_scores(
            self.predict_benchmark_logit_scores(capability_scores)
        )

    # loss due to the points that fall below the floor
    def intrinsic_loss(self) -> torch.Tensor:
        model_scores_floored = torch.max(self.train_model_scores, self.benchmark_floors)
        return F.mse_loss(self.train_model_scores, model_scores_floored)

    def forward(self, model_scores: torch.Tensor) -> torch.Tensor:
        logit_scores = self.predict_logit_scores(model_scores)
        capability_scores = self.predict_capability_scores(logit_scores)
        benchmark_logit_scores = self.predict_benchmark_logit_scores(capability_scores)
        return self.predict_benchmark_scores(benchmark_logit_scores)

    # compute loss
    @torch.compile(fullgraph=True, dynamic=True)
    def train_loss(self) -> torch.Tensor:
        return F.mse_loss(self.train_model_scores, self(self.train_model_scores))

    def fit(self, epochs=3000):
        optimizer = optim.Adam(params=self.parameters(), lr=1e-2, fused=True)
        for i in range(epochs):
            optimizer.zero_grad()
            l = self.train_loss()
            l.backward()
            optimizer.step()
            self.train_losses.append(l.item())
        self.eval()
