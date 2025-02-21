# %%
from dataclasses import dataclass

import duckdb
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import util_plot_markers
from util_linear_obs_scaling_law_predictor import LinearPC1Predictor
from util_logit_obs_scaling_law_predictor import LogitPC1Predictor


# Define the Chinchilla loss function parameter set
@dataclass
class ChinchillaParams:
    alpha: float
    beta: float
    A: float
    B: float
    E: float


# These numbers are from Hoffman et al. 2022
HOFF_PARAMS = ChinchillaParams(
    alpha=0.3392,
    beta=0.2849,
    A=406.4,
    B=410.7,
    E=1.6934,
)

# These numbers are from Epoch (Besiroglu et al. 2024)
EPOCH_PARAMS = ChinchillaParams(alpha=0.3478, beta=0.3658, A=482.01, B=2085.43, E=1.8172)


def loss(n, d, p: ChinchillaParams) -> float:
    return p.E + p.A / n**p.alpha + p.B / d**p.beta


def opt_params(L_budget: float, p: ChinchillaParams) -> tuple[float, float]:
    l = L_budget - p.E
    N_opt = (p.A * (p.alpha + p.beta) / (l * p.beta)) ** (1 / p.alpha)
    D_opt = (p.B * (p.alpha + p.beta) / (l * p.alpha)) ** (1 / p.beta)
    return N_opt, D_opt


base_llm_benchmark_eval = pd.read_csv("./data_models/meta/base_llm_benchmark_eval.csv")
family_release_dates = duckdb.read_csv("./data_models/meta/family_release_dates.csv")

base_llm_benchmark_eval = duckdb.sql(
    """
    SELECT 
        "Model",
        "Model Family",
        (year(release_date) + (1/365)*dayofyear(release_date)) as release_date,
        "MMLU", 
        "ARC-C", 
        "HellaSwag", 
        "Winograd", 
        "TruthfulQA", 
        "GSM8K", 
        "XWinograd", 
        "HumanEval", 
        "Model Size (B)", 
        "Pretraining Data Size (T)", 
        "FLOPs (1E21)"
    FROM base_llm_benchmark_eval
    JOIN family_release_dates ON base_llm_benchmark_eval."Model Family" = family_release_dates.family
    """
).df()

############################################
# Calculate Optimal Flops for each model
############################################

# add optimal params to the dataframe
for param, label in [(HOFF_PARAMS, "Hoffman"), (EPOCH_PARAMS, "Besiroglu")]:
    l_budgets = [
        loss(n * 1e9, d * 1e12, param)
        for n, d in zip(
            base_llm_benchmark_eval["Model Size (B)"],
            base_llm_benchmark_eval["Pretraining Data Size (T)"],
            strict=False,
        )
    ]
    n_opt, d_opt = zip(*[opt_params(l_budget, param) for l_budget in l_budgets], strict=False)
    base_llm_benchmark_eval[f"N_opt_{label}"] = n_opt
    base_llm_benchmark_eval[f"D_opt_{label}"] = d_opt
    base_llm_benchmark_eval[f"FLOPs_opt_{label} (1E21)"] = (
        6
        * base_llm_benchmark_eval[f"N_opt_{label}"]
        * base_llm_benchmark_eval[f"D_opt_{label}"]
        / 1e21
    )

# drop NaNs
base_llm_benchmark_eval.dropna(inplace=True)

# insert log flops
base_llm_benchmark_eval["log10 FLOPs (1E21)"] = np.log10(base_llm_benchmark_eval["FLOPs (1E21)"])
base_llm_benchmark_eval["log10 FLOPs_Hoffman (1E21)"] = np.log10(
    base_llm_benchmark_eval["FLOPs_opt_Hoffman (1E21)"]
)
base_llm_benchmark_eval["log10 FLOPs_opt_Besiroglu (1E21)"] = np.log10(
    base_llm_benchmark_eval["FLOPs_opt_Besiroglu (1E21)"]
)


############################################
# Calculate  Observational Capability Scores
############################################

benchmark_data = [
    ("MMLU", 0.25),
    ("ARC-C", 0.2),
    ("HellaSwag", 0.25),
    ("Winograd", 0.5),
    # ("TruthfulQA", 0.4),
    ("GSM8K", 0.0),
    ("XWinograd", 0.5),
    ("HumanEval", 0.0),
]

benchmarks, benchmark_floor = zip(*benchmark_data, strict=False)
benchmarks = list(benchmarks)

model_scores = torch.tensor(base_llm_benchmark_eval[benchmarks].values, dtype=torch.float32)

############################################
# Calculate Logit Observational Capability Scores
############################################

logit_obs_model = LogitPC1Predictor(benchmarks, benchmark_floor, model_scores)
logit_obs_model.fit()

base_llm_benchmark_eval["PC-1"] = (
    logit_obs_model.predict_capability_scores_from_model_scores(model_scores).detach().numpy()
)


############################################
# Calculate Linear Observational Capability Scores
############################################

linear_obs_model = LinearPC1Predictor(benchmarks, benchmark_floor, model_scores)
linear_obs_model.fit()

base_llm_benchmark_eval["Linear PC-1"] = (
    linear_obs_model.predict_capability_scores_from_model_scores(model_scores).detach().numpy()
)

# %%
############################################
# Plot each model family on the same graph
# Except, color each line based on time so we can see the progression
############################################
families_release_dates = (
    base_llm_benchmark_eval[["Model Family", "release_date"]].drop_duplicates().values
)

families, release_dates = zip(*families_release_dates, strict=False)

fig, ax = plt.subplots(1, 1, figsize=(14, 14))  # 1 columns

ax.set_title("FLOPs_opt vs PC-1")
ax.set_xlabel("log10 FLOPs_opt (1E21)")
ax.set_ylabel("PC-1")
norm = mcolors.Normalize(vmin=min(release_dates), vmax=max(release_dates))
cmap = plt.get_cmap("viridis")
fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, label="Release Date")

linfit_release_dates = []
linfit_yintercepts = []
linfit_slopes = []

for i, (family, release_date) in enumerate(families_release_dates):
    print("release_date", release_date)
    xpoints = base_llm_benchmark_eval[base_llm_benchmark_eval["Model Family"] == family][
        "log10 FLOPs_opt_Besiroglu (1E21)"
    ]

    ypoints = base_llm_benchmark_eval[base_llm_benchmark_eval["Model Family"] == family]["PC-1"]
    # Fit a line to the data
    lparams = np.polyfit(xpoints, ypoints, 1)

    # # discard outliers
    # if lparams[0] < 0:
    #     continue

    # add the line fit to the list
    linfit_release_dates.append(release_date)
    linfit_yintercepts.append(lparams[1])
    linfit_slopes.append(lparams[0])

    # plot
    xspace = np.linspace(xpoints.min(), xpoints.max(), 100)
    y_line = np.polyval(lparams, xspace)
    # compute MSE
    linear_mse = np.mean((ypoints - np.polyval(lparams, xpoints)) ** 2)

    ax.scatter(xpoints, ypoints, marker=util_plot_markers.markers[i], label=family)
    ax.plot(xspace, y_line, label="Linear", color=cmap(norm(release_date)))

ax.legend()

fig, ax = plt.subplots(2, 1, figsize=(14, 14))  # 1 columns

ax[0].set_title("Slope of Linear Fit vs Release Date")
ax[0].set_xlabel("Release Date")
ax[0].set_ylabel("Slope")
ax[0].scatter(linfit_release_dates, linfit_slopes)

# fit line
x = np.array(linfit_release_dates)
y = np.array(linfit_slopes)
m, b = np.polyfit(x, y, 1)
ax[0].plot(x, m * x + b, label=f"y = {m:.2f}x + {b:.2f}")
ax[0].legend()


ax[1].set_title("Y-intercept of Linear Fit vs Release Date")
ax[1].set_xlabel("Release Date")
ax[1].set_ylabel("Y-intercept")
ax[1].scatter(linfit_release_dates, linfit_yintercepts)

# fit line
x = np.array(linfit_release_dates)
y = np.array(linfit_yintercepts)
algprog_m, algprog_b = np.polyfit(x, y, 1)
ax[1].plot(x, algprog_m * x + algprog_b, label=f"y = {algprog_m:.2f}x + {algprog_b:.2f}")
ax[1].legend()

# %%

# Determine how fast model frontier flops_opt are increasing over time
# Compute de-algorithmically-progressed improvement rate
# Compute the average improvement rate of the model frontier


fig, ax = plt.subplots(1, 1, figsize=(14, 14))  # 1 columns

ax.set_title("FLOPs_opt vs deprogressed PC-1")
ax.set_xlabel("log10 FLOPs_opt (1E21)")
ax.set_ylabel("deprogressed PC-1")
norm = mcolors.Normalize(vmin=min(release_dates), vmax=max(release_dates))
cmap = plt.get_cmap("viridis")
fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, label="Release Date")

x = np.array(linfit_release_dates)
y = np.array(linfit_yintercepts)
m, b = np.polyfit(x, y, 1)

model_pc1_deprogressed = []

for i, (family, release_date) in enumerate(families_release_dates):
    print("release_date", release_date)
    xpoints = base_llm_benchmark_eval[base_llm_benchmark_eval["Model Family"] == family][
        "log10 FLOPs_opt_Besiroglu (1E21)"
    ]

    ypoints = base_llm_benchmark_eval[base_llm_benchmark_eval["Model Family"] == family]["PC-1"]

    # de-algorithmic-progress the model family y axis
    algorithmic_progress_boost = m * release_date + b
    ypoints = ypoints - algorithmic_progress_boost

    # add the de-algorithmic-progressed data to the list
    model_pc1_deprogressed.extend(ypoints)

    # Fit a line to the data
    lparams = np.polyfit(xpoints, ypoints, 1)

    # plot
    xspace = np.linspace(xpoints.min(), xpoints.max(), 100)
    y_line = np.polyval(lparams, xspace)
    # compute MSE
    linear_mse = np.mean((ypoints - np.polyval(lparams, xpoints)) ** 2)

    ax.scatter(xpoints, ypoints, marker=util_plot_markers.markers[i], label=family)
    ax.plot(xspace, y_line, label="Linear", color=cmap(norm(release_date)))

ax.legend()

base_llm_benchmark_eval["PC-1_deprogressed"] = model_pc1_deprogressed

# plot the de-algorithmic-progressed model vs flops_opt
fig, ax = plt.subplots(1, 1, figsize=(14, 14))  # 1 columns

ax.set_title("FLOPs_opt vs deprogressed PC-1")
ax.set_xlabel("log10 FLOPs_opt (1E21)")
ax.set_ylabel("deprogressed PC-1")
ax.scatter(
    base_llm_benchmark_eval["log10 FLOPs_opt_Besiroglu (1E21)"],
    base_llm_benchmark_eval["PC-1_deprogressed"],
)


# fit line
dealgprog_m, dealgprog_b = np.polyfit(
    base_llm_benchmark_eval["log10 FLOPs_opt_Besiroglu (1E21)"],
    base_llm_benchmark_eval["PC-1_deprogressed"],
    1,
)

xspace = np.linspace(
    base_llm_benchmark_eval["log10 FLOPs_opt_Besiroglu (1E21)"].min(),
    base_llm_benchmark_eval["log10 FLOPs_opt_Besiroglu (1E21)"].max(),
    100,
)
y_line = np.polyval([dealgprog_m, dealgprog_b], xspace)

ax.plot(
    xspace,
    y_line,
    label=f"y = {dealgprog_m:.2f}x + {dealgprog_b:.2f}",
)

ax.legend()

# select only frontier models
frontier_llm_benchmark_eval = duckdb.sql(
    """
    SELECT b1.*
    FROM base_llm_benchmark_eval b1
    WHERE b1."log10 FLOPs_opt_Besiroglu (1E21)" >= (
        SELECT b2."log10 FLOPs_opt_Besiroglu (1E21)"
        FROM base_llm_benchmark_eval b2
        WHERE b2.release_date <= b1.release_date
        ORDER BY b2."log10 FLOPs_opt_Besiroglu (1E21)" DESC
        LIMIT 1
    )
    """
).df()

# get frontier model progression rate (in flops_opt)
fig, ax = plt.subplots(1, 1, figsize=(14, 14))  # 1 columns

ax.set_title("Frontier Model Progression Rate")
ax.set_xlabel("Release Date")
ax.set_ylabel("FLOPs_opt (1E21)")


ax.scatter(
    base_llm_benchmark_eval["release_date"],
    base_llm_benchmark_eval["log10 FLOPs_opt_Besiroglu (1E21)"],
    color="blue",
)
ax.scatter(
    frontier_llm_benchmark_eval["release_date"],
    frontier_llm_benchmark_eval["log10 FLOPs_opt_Besiroglu (1E21)"],
    color="green",
)

flops_opt_vs_rd_m, flops_opt_vs_rd_b = np.polyfit(
    frontier_llm_benchmark_eval["release_date"],
    frontier_llm_benchmark_eval["log10 FLOPs_opt_Besiroglu (1E21)"],
    1,
)

xspace = np.linspace(
    base_llm_benchmark_eval["release_date"].min(),
    base_llm_benchmark_eval["release_date"].max(),
    100,
)
y_line = np.polyval([flops_opt_vs_rd_m, flops_opt_vs_rd_b], xspace)

ax.plot(
    xspace,
    y_line,
    label=f"y = {flops_opt_vs_rd_m:.2f}x + {flops_opt_vs_rd_b:.2f}",
)

ax.legend()

# %%

############################
# Expected Frontier FLOPs_opt and PC-1 vs Release Date
############################


# Create function that takes in release date, and computes the expected FLOPs_opt, and PC-1
def expected_flops_opt(release_date: np.ndarray) -> np.ndarray:
    return flops_opt_vs_rd_m * release_date + flops_opt_vs_rd_b


def expected_pc1(log10_flops_opt: np.ndarray, release_date: np.ndarray) -> np.ndarray:
    dealgprog_pc1 = dealgprog_m * log10_flops_opt + dealgprog_b
    algorithmic_progress_boost = algprog_m * release_date + algprog_b
    return dealgprog_pc1 + algorithmic_progress_boost


fig, ax = plt.subplots(2, 1, figsize=(14, 14))  # 1 columns

# Plot the expected frontier FLOPs_opt vs release date
ax[0].set_title("Release Date vs Expected Frontier FLOPs_opt")
ax[0].set_xlabel("Release Date")
ax[0].set_ylabel("log10 FLOPs_opt (1E21)")

xspace = np.linspace(
    base_llm_benchmark_eval["release_date"].min(),
    base_llm_benchmark_eval["release_date"].max(),
    100,
)

y_line = expected_flops_opt(xspace)

ax[0].plot(
    xspace,
    y_line,
    label="Expected log10 FLOPs_opt",
)

ax[0].scatter(
    base_llm_benchmark_eval["release_date"],
    base_llm_benchmark_eval["log10 FLOPs_opt_Besiroglu (1E21)"],
    color="blue",
)

ax[0].scatter(
    frontier_llm_benchmark_eval["release_date"],
    frontier_llm_benchmark_eval["log10 FLOPs_opt_Besiroglu (1E21)"],
    color="green",
    label="Frontier PC-1",
)


# Plot the expected frontier PC-1 vs release date
ax[1].set_title("Release Date vs Expected Frontier PC-1")
ax[1].set_xlabel("Release Date")
ax[1].set_ylabel("PC-1")

xspace = np.linspace(
    base_llm_benchmark_eval["release_date"].min(),
    base_llm_benchmark_eval["release_date"].max(),
    100,
)

y_line = expected_pc1(expected_flops_opt(xspace), xspace)

ax[1].plot(
    xspace,
    y_line,
    label="Expected PC-1",
)

ax[1].scatter(
    base_llm_benchmark_eval["release_date"],
    base_llm_benchmark_eval["PC-1"],
    color="blue",
)

ax[1].scatter(
    frontier_llm_benchmark_eval["release_date"],
    frontier_llm_benchmark_eval["PC-1"],
    color="green",
    label="Frontier PC-1",
)
ax[1].legend()

############################
# MMLU vs FLOPs_opt at different release dates
############################

fig, ax = plt.subplots(1, 1, figsize=(14, 14))  # 1 columns

ax.set_title("MMLU vs FLOPs_opt")
ax.set_xlabel("log10 FLOPs_opt (1E21)")
ax.set_ylabel("MMLU")
norm = mcolors.Normalize(vmin=min(release_dates), vmax=max(release_dates))
cmap = plt.get_cmap("viridis")
fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, label="Release Date")

for i, (family, release_date) in enumerate(families_release_dates):
    print("release_date", release_date)
    xpoints = base_llm_benchmark_eval[base_llm_benchmark_eval["Model Family"] == family][
        "log10 FLOPs_opt_Besiroglu (1E21)"
    ]

    ypoints = base_llm_benchmark_eval[base_llm_benchmark_eval["Model Family"] == family]["MMLU"]

    ax.scatter(
        xpoints,
        ypoints,
        marker=util_plot_markers.markers[i],
        label=family,
        color=cmap(norm(release_date)),
    )

for release_date in [2021.5, 2022.5, 2023.5, 2024.5]:
    xspace = np.linspace(
        base_llm_benchmark_eval["log10 FLOPs_opt_Besiroglu (1E21)"].min(),
        base_llm_benchmark_eval["log10 FLOPs_opt_Besiroglu (1E21)"].max(),
        100,
    )

    pc1 = expected_pc1(xspace, np.full_like(xspace, release_date))
    mmlu_score = (
        logit_obs_model.predict_benchmark_scores(
            logit_obs_model.predict_benchmark_logit_scores(torch.tensor(pc1, dtype=torch.float32))
        )
        .T[0]
        .detach()
        .numpy()
    )

    ax.plot(
        xspace,
        mmlu_score,
        label=f"Expected MMLU {release_date}",
        color=cmap(norm(release_date)),
    )

ax.legend()

############################
# Linear PC-1 vs FLOPs_opt at different release dates
############################

fig, ax = plt.subplots(1, 1, figsize=(14, 14))  # 1 columns

ax.set_title("Linear PC-1 vs FLOPs_opt")
ax.set_xlabel("log10 FLOPs_opt (1E21)")
ax.set_ylabel("Linear PC-1")
norm = mcolors.Normalize(vmin=min(release_dates), vmax=max(release_dates))
cmap = plt.get_cmap("viridis")
fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, label="Release Date")

for i, (family, release_date) in enumerate(families_release_dates):
    print("release_date", release_date)
    xpoints = base_llm_benchmark_eval[base_llm_benchmark_eval["Model Family"] == family][
        "log10 FLOPs_opt_Besiroglu (1E21)"
    ]

    ypoints = base_llm_benchmark_eval[base_llm_benchmark_eval["Model Family"] == family][
        "Linear PC-1"
    ]

    ax.scatter(
        xpoints,
        ypoints,
        marker=util_plot_markers.markers[i],
        label=family,
        color=cmap(norm(release_date)),
    )

for release_date in [2021.5, 2022.5, 2023.5, 2024.5]:
    xspace = np.linspace(
        base_llm_benchmark_eval["log10 FLOPs_opt_Besiroglu (1E21)"].min(),
        base_llm_benchmark_eval["log10 FLOPs_opt_Besiroglu (1E21)"].max(),
        100,
    )

    pc1 = expected_pc1(xspace, np.full_like(xspace, release_date))
    pred_benchmark_scores = (
        logit_obs_model.predict_benchmark_scores(
            logit_obs_model.predict_benchmark_logit_scores(torch.tensor(pc1, dtype=torch.float32))
        )
        .detach()
        .numpy()
    )

    pred_linear_pc1 = (
        linear_obs_model.predict_capability_scores_from_model_scores(
            torch.tensor(pred_benchmark_scores, dtype=torch.float32)
        )
        .detach()
        .numpy()
    )

    ax.plot(
        xspace,
        pred_linear_pc1,
        label=f"Expected PC-1 {release_date}",
        color=cmap(norm(release_date)),
    )

ax.legend()
