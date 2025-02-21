from typing import override

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from util_obs_scaling_law_predictor import ObsScalingLawPredictor

PC1_EPS = 1e-4


class LinearPC1Predictor(ObsScalingLawPredictor):
    @override
    def __init__(
        self,
        benchmarks: list[str],
        benchmark_floors: list[float],
        train_model_scores: torch.Tensor,
        standardize=False,
    ):
        super().__init__(
            benchmarks,
            benchmark_floors,
            train_model_scores,
        )
        self.standardize = standardize

        self.train_losses = []

        B = len(benchmarks)
        self.benchmarks = benchmarks
        self.intermediate = "PC-1"

        # in M x B
        self.train_model_scores = nn.Buffer(train_model_scores)

        # Note: We initialize with values that are likely to be close to the true values, but the model will learn them in training

        # in B
        self.benchmark_weights = self.pca_benchmark_weights
        self.alpha = nn.Parameter(torch.zeros(B, dtype=torch.float32))
        self.beta = nn.Parameter(torch.full((B,), fill_value=0.5, dtype=torch.float32))

    @property
    def pca_benchmark_weights(self) -> torch.Tensor:
        if self.standardize:
            # standardize the scores
            train_model_scores = (
                self.train_model_scores - self.train_model_scores.mean(dim=0)
            ) / (self.train_model_scores.std(dim=0) + PC1_EPS)
        else:
            train_model_scores = self.train_model_scores

        # perform PCA and get the first component, return it
        # logit_scores: M x B
        # benchmark_weights: B x 1
        U, S, V = torch.pca_lowrank(train_model_scores)
        # gamma in B x 1

        # flip all weights if V is all negative
        weights = V[:, 0]
        if torch.all(weights < 0):
            return -weights
        else:
            return weights

    @override
    def predict_capability_scores_from_model_scores(
        self, model_scores: torch.Tensor
    ) -> torch.Tensor:
        # benchmark_weights: B x 1
        benchmark_weights = self.benchmark_weights.unsqueeze(1)
        # benchmark_weights = self.benchmark_weights.unsqueeze(1)
        # M x 1
        capability_score = model_scores @ benchmark_weights
        return capability_score.squeeze(1)

    @override
    def predict_benchmark_scores_from_capability_scores(
        self, capability_scores: torch.Tensor
    ) -> torch.Tensor:
        # capability_scores: M x 1
        capability_scores = capability_scores.unsqueeze(1)
        # beta = 1 x B
        beta = self.beta.unsqueeze(0)
        # M x B
        predicted_logit_scores = capability_scores @ beta + self.alpha
        return predicted_logit_scores

    # compute loss
    def forward(self, model_scores: torch.Tensor) -> torch.Tensor:
        capability_scores = self.predict_capability_scores_from_model_scores(
            model_scores
        )
        return self.predict_benchmark_scores_from_capability_scores(capability_scores)

    # compute loss
    @torch.compile(fullgraph=True, dynamic=True)
    def train_loss(self) -> torch.Tensor:
        return F.mse_loss(
            self.train_model_scores, self.forward(self.train_model_scores)
        )

    def fit(self, epochs=500):
        optimizer = optim.Adam(params=self.parameters(), lr=1e-1, fused=True)
        for i in range(epochs):
            optimizer.zero_grad()
            l = self.train_loss()
            l.backward()
            optimizer.step()
            self.train_losses.append(l.item())
