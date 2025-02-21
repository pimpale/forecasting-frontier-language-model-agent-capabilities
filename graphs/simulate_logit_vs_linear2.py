# %%

from util_linear_obs_scaling_law_predictor import LinearPC1Predictor
from util_logit_obs_scaling_law_predictor import LogitPC1Predictor
from util_obs_scaling_law_predictor import ScalingLaw
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim

N_MODELS = 1000


@dataclass
class Benchmark:
    """
    Score = sigmoid(beta*(x - alpha))
    """

    name: str
    alpha: float
    beta: float
    noise: float = 0.0


@dataclass
class Case:
    """
    Test Case
    """

    benchmarks: list[Benchmark]
    to_predict: Benchmark


def gen_synthetic_data(
    case: Case, capability_scores: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic data for a given test case
    """
    benchmark_scores = torch.zeros(N_MODELS, len(case.benchmarks), dtype=torch.float32)

    to_predict = torch.zeros(N_MODELS, dtype=torch.float32)

    for i, bm in enumerate(case.benchmarks + [case.to_predict]):
        noise = torch.randn(N_MODELS) * bm.noise
        result = torch.sigmoid(bm.beta * (capability_scores - bm.alpha) + noise)
        if bm == case.to_predict:
            to_predict = result
        else:
            benchmark_scores[:, i] = result

    return benchmark_scores, to_predict


def train_models(
    case: Case, benchmark_scores: torch.Tensor, to_predict_scores: torch.Tensor,
    standardize=False,
) -> tuple[LinearPC1Predictor, LogitPC1Predictor, ScalingLaw, ScalingLaw]:
    """
    Train Linear and Logit PC1 models
    """
    linpc1 = LinearPC1Predictor(
        benchmarks=[bm.name for bm in case.benchmarks],
        benchmark_floors=[0.0 for bm in case.benchmarks],
        train_model_scores=benchmark_scores,
        standardize=standardize,
    )
    print("Fitting Linear PC1")
    linpc1.fit()

    pred_capability = linpc1.predict_capability_scores_from_model_scores(
        benchmark_scores
    ).detach()

    linslaw = ScalingLaw(
        benchmark=case.to_predict.name,
        floor=0.0,
        maybe_ceil=1.0,
        capability_scores=pred_capability,
        benchmark_scores=to_predict_scores,
    )
    print("Fitting Linear Scaling Law")
    linslaw.fit()

    logpc1 = LogitPC1Predictor(
        benchmarks=[bm.name for bm in case.benchmarks],
        benchmark_floors=[0.0 for bm in case.benchmarks],
        train_model_scores=benchmark_scores,
    )
    print("Fitting Logit PC1")
    logpc1.fit()

    pred_capability = logpc1.predict_capability_scores_from_model_scores(
        benchmark_scores
    ).detach()

    logslaw = ScalingLaw(
        benchmark=case.to_predict.name,
        floor=0.0,
        maybe_ceil=1.0,
        capability_scores=pred_capability,
        benchmark_scores=to_predict_scores,
    )
    print("Fitting Logit Scaling Law")
    logslaw.fit()

    return linpc1, logpc1, linslaw, logslaw


def plot_pca_components(
    data: torch.Tensor,
    xlabel: str = "X",
    ylabel: str = "Y",
    base_scale: float = 0.02,
    x_plot_range: tuple = (0, 1),
    y_plot_range: tuple = (0, 1),
    equal_aspect: bool = True,
):
    """
    Plot PCA components for a 2D tensor with arrows showing principal directions.

    Args:
        data: 2D tensor of shape (n_samples, 2)
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        base_scale: Scale factor for PCA arrows
        arrow_head_width: Width of arrow heads
        arrow_head_length: Length of arrow heads
        plot_range: Tuple of (min, max) for both axes
    """

    scatter = plt.scatter(data[:, 0], data[:, 1], alpha=0.1, color="tab:blue")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    mean_point = torch.mean(data, dim=0)
    U, S, V = torch.pca_lowrank(data)

    # Ensure consistent direction (point up/right)
    if torch.all(V[:, 0] < 0):
        V[:, 0] = -V[:, 0]
    if torch.all(V[:, 1] < 0):
        V[:, 1] = -V[:, 1]

    # Calculate explained variance percentages
    pc1_var = (S[0] ** 2 / torch.sum(S**2) * 100).item()
    pc2_var = (S[1] ** 2 / torch.sum(S**2) * 100).item()

    # Plot arrows for principal components using annotate
    plt.annotate(
        "",
        xy=(
            mean_point[0].item() + base_scale * S[0].item() * V[0, 0].item(),
            mean_point[1].item() + base_scale * S[0].item() * V[1, 0].item(),
        ),
        xytext=(mean_point[0].item(), mean_point[1].item()),
        arrowprops=dict(facecolor="tab:orange", shrink=0),
    )

    plt.annotate(
        "",
        xy=(
            mean_point[0].item() + base_scale * S[1].item() * V[0, 1].item(),
            mean_point[1].item() + base_scale * S[1].item() * V[1, 1].item(),
        ),
        xytext=(mean_point[0].item(), mean_point[1].item()),
        arrowprops=dict(facecolor="tab:green", shrink=0),
    )

    plt.xlim(*x_plot_range)
    plt.ylim(*y_plot_range)
    if equal_aspect:
        plt.gca().set_aspect("equal")

    # Add legend manually
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="tab:blue",
            alpha=0.1,
            markersize=10,
            label="Data Points",
            linestyle="None",
        ),
        Patch(facecolor="tab:orange", label=f"PC1 ({pc1_var:.1f}% var)"),
        Patch(facecolor="tab:green", label=f"PC2 ({pc2_var:.1f}% var)"),
    ]
    plt.legend(handles=legend_elements)


def normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    """
    Normalize scores to [0, 1] range
    """
    return (scores - torch.min(scores)) / (torch.max(scores) - torch.min(scores))


def normalize_scores2(scores: torch.Tensor, other_scores: torch.Tensor) -> torch.Tensor:
    """
    Normalize scores to [0, 1] range
    """
    all_scores = torch.cat([scores, other_scores])
    return (scores - torch.min(all_scores)) / (
        torch.max(all_scores) - torch.min(all_scores)
    )


# %%

# Edge Case 1
ec1 = Case(
    benchmarks=[Benchmark("A", 10.0, 0.1, 0.01), Benchmark("B", 90.0, 0.1, 0.00)],
    to_predict=Benchmark("C", 50.0, 0.1, 0.00),
)

# Generate Model latent abilities
# Uniform distribution from 0 to 100
model_latent_capability_ec1 = torch.linspace(
    0, 100, steps=N_MODELS, dtype=torch.float32
)

# Generate Benchmarks scores from model latent abilities using Item Response Theory
benchmark_scores_ec1, to_predict_scores_ec1 = gen_synthetic_data(
    ec1, model_latent_capability_ec1
)

# Train Linear and Logit PC1 models
linpc1_ec1, logpc1_ec1, linslaw_ec1, logslaw_ec1 = train_models(
    ec1, benchmark_scores_ec1, to_predict_scores_ec1
)


# %%
plt.plot(model_latent_capability_ec1, benchmark_scores_ec1[:, 0], label="Benchmark A")
plt.plot(model_latent_capability_ec1, benchmark_scores_ec1[:, 1], label="Benchmark B")
plt.plot(model_latent_capability_ec1, to_predict_scores_ec1, label="Benchmark C")
plt.xlabel("Model Latent Capability")
plt.ylabel("Benchmark Scores")
plt.legend()
plt.show()

# %%

linpc1_cap = linpc1_ec1.predict_capability_scores_from_model_scores(
    benchmark_scores_ec1
)
logpc1_cap = logpc1_ec1.predict_capability_scores_from_model_scores(
    benchmark_scores_ec1
).detach()

plt.plot(model_latent_capability_ec1, normalize_scores(linpc1_cap), label="Linear PC1")
plt.plot(model_latent_capability_ec1, normalize_scores(logpc1_cap), label="Logit PC1")
plt.xlabel("Model Latent Capability")
plt.ylabel("Capability Scores")
plt.title("Linear vs Logit PC1")
plt.legend()

# %%

linpc1_pred = linslaw_ec1.forward(linpc1_cap).detach()
logpc1_pred = logslaw_ec1.forward(logpc1_cap).detach()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Left subplot: Predictions
ax1.plot(
    model_latent_capability_ec1,
    to_predict_scores_ec1,
    linestyle="--",
    color="black",
    label="True Benchmark C",
)
ax1.plot(model_latent_capability_ec1, linpc1_pred, label="Linear PC1 Prediction")
ax1.plot(model_latent_capability_ec1, logpc1_pred, label="Logit PC1 Prediction")
ax1.set_title("Predictions of Benchmark C")
ax1.set_xlabel("Model Latent Capability")
ax1.set_ylabel("Benchmark Scores")
ax1.legend()

# Right subplot: Errors
ax2.plot(
    model_latent_capability_ec1,
    torch.zeros_like(model_latent_capability_ec1),
    linestyle="--",
    color="black",
)
ax2.plot(
    model_latent_capability_ec1, linpc1_pred - to_predict_scores_ec1, label="Linear PC1"
)
ax2.plot(
    model_latent_capability_ec1, logpc1_pred - to_predict_scores_ec1, label="Logit PC1"
)
ax2.set_title("Prediction Errors")
ax2.set_xlabel("Model Latent Capability")
ax2.set_ylabel("Prediction Error")
ax2.legend()

plt.tight_layout()
plt.show()


# %%

# Edge Case 2
ec2 = Case(
    benchmarks=[Benchmark("A", 10.0, 0.1, 0.01), Benchmark("B", 90.0, 0.1, 0.00)],
    to_predict=Benchmark("C", 50.0, 0.1, 0.00),
)

# Generate Model latent abilities
# Uniform distribution from 0 to 100
train_model_latent_capability_ec2 = torch.linspace(
    0, 50, steps=N_MODELS, dtype=torch.float32
)
test_model_latent_capability_ec2 = torch.linspace(
    50, 100, steps=N_MODELS, dtype=torch.float32
)

# Generate Benchmarks scores from model latent abilities using Item Response Theory
train_benchmark_scores_ec2, train_to_predict_scores_ec2 = gen_synthetic_data(
    ec2, train_model_latent_capability_ec2
)
test_benchmark_scores_ec2, test_to_predict_scores_ec2 = gen_synthetic_data(
    ec2, test_model_latent_capability_ec2
)

# Train Linear and Logit PC1 models
linpc1_ec2, logpc1_ec2, linslaw_ec2, logslaw_ec2 = train_models(
    ec2, train_benchmark_scores_ec2, train_to_predict_scores_ec2,
    standardize=True
)

# %%

plt.plot(
    train_model_latent_capability_ec2,
    train_benchmark_scores_ec2[:, 0],
    color="C0",
    label="Train Benchmark A",
)
plt.plot(
    train_model_latent_capability_ec2,
    train_benchmark_scores_ec2[:, 1],
    color="C1",
    label="Train Benchmark B",
)
plt.plot(
    train_model_latent_capability_ec2,
    train_to_predict_scores_ec2,
    color="C2",
    label="Train Benchmark C",
)
plt.plot(
    test_model_latent_capability_ec2,
    test_benchmark_scores_ec2[:, 0],
    color="C0",
    linestyle="--",
    label="Test Benchmark A",
)
plt.plot(
    test_model_latent_capability_ec2,
    test_benchmark_scores_ec2[:, 1],
    color="C1",
    linestyle="--",
    label="Test Benchmark B",
)
plt.plot(
    test_model_latent_capability_ec2,
    test_to_predict_scores_ec2,
    color="C2",
    linestyle="--",
    label="Test Benchmark C",
)
plt.xlabel("Model Latent Capability")
plt.ylabel("Benchmark Scores")
plt.legend()

# %%
linpc1_cap_train = linpc1_ec2.predict_capability_scores_from_model_scores(
    train_benchmark_scores_ec2
)
logpc1_cap_train = logpc1_ec2.predict_capability_scores_from_model_scores(
    train_benchmark_scores_ec2
).detach()
linpc1_cap_test = linpc1_ec2.predict_capability_scores_from_model_scores(
    test_benchmark_scores_ec2
)
logpc1_cap_test = logpc1_ec2.predict_capability_scores_from_model_scores(
    test_benchmark_scores_ec2
).detach()


plt.plot(
    train_model_latent_capability_ec2,
    normalize_scores2(linpc1_cap_train, linpc1_cap_test),
    color="C0",
    label="Linear PC1",
)
plt.plot(
    train_model_latent_capability_ec2,
    normalize_scores2(logpc1_cap_train, logpc1_cap_test),
    color="C1",
    label="Logit PC1",
)
plt.plot(
    test_model_latent_capability_ec2,
    normalize_scores2(linpc1_cap_test, linpc1_cap_train),
    linestyle="dashed",
    color="C0",
    label="Linear PC1",
)
plt.plot(
    test_model_latent_capability_ec2,
    normalize_scores2(logpc1_cap_test, logpc1_cap_train),
    linestyle="dashed",
    color="C1",
    label="Logit PC1",
)
plt.xlabel("Model Latent Capability")
plt.ylabel("Capability Scores")
plt.title("Linear vs Logit PC1")
plt.legend()
plt.show()


# %%

# Get predictions for both train and test sets
linpc1_pred_train = linslaw_ec2.forward(linpc1_cap_train).detach()
logpc1_pred_train = logslaw_ec2.forward(logpc1_cap_train).detach()
linpc1_pred_test = linslaw_ec2.forward(linpc1_cap_test).detach()
logpc1_pred_test = logslaw_ec2.forward(logpc1_cap_test).detach()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Left subplot: Predictions
ax1.plot(
    train_model_latent_capability_ec2,
    train_to_predict_scores_ec2,
    linestyle="--",
    color="black",
    label="True Benchmark C (Train)",
)
ax1.plot(
    test_model_latent_capability_ec2,
    test_to_predict_scores_ec2,
    linestyle=":",
    color="black",
    label="True Benchmark C (Test)",
)
ax1.plot(
    train_model_latent_capability_ec2,
    linpc1_pred_train,
    color="C0",
    label="Linear PC1 (Train)",
)
ax1.plot(
    train_model_latent_capability_ec2,
    logpc1_pred_train,
    color="C1",
    label="Logit PC1 (Train)",
)
ax1.plot(
    test_model_latent_capability_ec2,
    linpc1_pred_test,
    color="C0",
    linestyle="--",
    label="Linear PC1 (Test)",
)
ax1.plot(
    test_model_latent_capability_ec2,
    logpc1_pred_test,
    color="C1",
    linestyle="--",
    label="Logit PC1 (Test)",
)
ax1.set_title("Predictions of Benchmark C")
ax1.set_xlabel("Model Latent Capability")
ax1.set_ylabel("Benchmark Scores")
ax1.legend()

# Right subplot: Errors
ax2.plot(
    torch.cat([train_model_latent_capability_ec2, test_model_latent_capability_ec2]),
    torch.zeros_like(
        torch.cat([train_model_latent_capability_ec2, test_model_latent_capability_ec2])
    ),
    linestyle="--",
    color="black",
)
ax2.plot(
    train_model_latent_capability_ec2,
    linpc1_pred_train - train_to_predict_scores_ec2,
    color="C0",
    label="Linear PC1 (Train)",
)
ax2.plot(
    train_model_latent_capability_ec2,
    logpc1_pred_train - train_to_predict_scores_ec2,
    color="C1",
    label="Logit PC1 (Train)",
)
ax2.plot(
    test_model_latent_capability_ec2,
    linpc1_pred_test - test_to_predict_scores_ec2,
    color="C0",
    linestyle="--",
    label="Linear PC1 (Test)",
)
ax2.plot(
    test_model_latent_capability_ec2,
    logpc1_pred_test - test_to_predict_scores_ec2,
    color="C1",
    linestyle="--",
    label="Logit PC1 (Test)",
)
ax2.set_title("Prediction Errors")
ax2.set_xlabel("Model Latent Capability")
ax2.set_ylabel("Prediction Error")
ax2.legend()

plt.tight_layout()
plt.show()

# %%

theta = torch.tensor(0.45)
rot_mat = torch.tensor(
    [[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]]
)
scale_mat = torch.eye(2) * torch.tensor([0.15, 0.05])

pca_demo_data = torch.distributions.MultivariateNormal(
    torch.tensor([0.5, 0.5]),
    covariance_matrix=rot_mat @ scale_mat @ scale_mat.T @ rot_mat.T,
).sample((1000,))

plot_pca_components(
    pca_demo_data, xlabel="Benchmark A", ylabel="Benchmark B", base_scale=0.1
)

# %%

plot_pca_components(
    train_benchmark_scores_ec2,
    xlabel="Benchmark A",
    ylabel="Benchmark B",
    base_scale=0.03,
    x_plot_range=(0.23, 1),
    y_plot_range=(0, 0.05),
    equal_aspect=False,
)

# %%

standardized_train_benchmark_scores_ec2 = (
    train_benchmark_scores_ec2 - torch.mean(train_benchmark_scores_ec2, dim=0)
) / torch.std(train_benchmark_scores_ec2, dim=0)

plot_pca_components(
    standardized_train_benchmark_scores_ec2,
    xlabel="Benchmark A",
    ylabel="Benchmark B",
    base_scale=0.03,
    x_plot_range=(-2.5, 2),
    y_plot_range=(-1, 3.5),
    equal_aspect=False,
)

# %%