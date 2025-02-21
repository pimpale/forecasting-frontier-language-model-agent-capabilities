# %%
import time
from collections import defaultdict
from dataclasses import dataclass

import duckdb
import matplotlib
import matplotlib.axes
import matplotlib.cm
import matplotlib.colorbar
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
import torch
import torch._dynamo.cache_size
import torch.nn.functional as F
from util_direct_date_predictor import DirectDatePredictor
from util_direct_elo_predictor import DirectEloPredictor
from util_direct_flop_predictor import DirectLogFlopPredictor
from util_linear_obs_scaling_law_predictor import LinearPC1Predictor
from util_obs_scaling_law_predictor import ObsScalingLawPredictor, ScalingLaw
from util_timeseries_backtesting import (
    BacktestSplitter,
    ExpandingWindowBacktestSplitter,
)

torch.set_num_threads(1)
torch._dynamo.cache_size.config.cache_size_limit = 1e9


# Define the Chinchilla loss function parameter set
@dataclass
class ChinchillaParams:
    alpha: float
    beta: float
    A: float
    B: float
    E: float


# These numbers are from Epoch (Besiroglu et al. 2024)
EPOCH_PARAMS = ChinchillaParams(alpha=0.3478, beta=0.3658, A=482.01, B=2085.43, E=1.8172)


def loss(n: float, d: float, p: ChinchillaParams) -> float:
    return p.E + p.A / n**p.alpha + p.B / d**p.beta


def opt_params(L_budget: float, p: ChinchillaParams) -> tuple[float, float]:
    l = L_budget - p.E
    N_opt = (p.A * (p.alpha + p.beta) / (l * p.beta)) ** (1 / p.alpha)
    D_opt = (p.B * (p.alpha + p.beta) / (l * p.alpha)) ** (1 / p.beta)
    return N_opt, D_opt


# add flops and optimal flops to the dataframe
def augment_df_opt_flops(
    df: pd.DataFrame,
):
    # insert flops
    df["FLOP (1E21)"] = 6 * df["N"] * df["D"]
    # insert log flops
    df["log10 FLOP"] = np.log10(df["FLOP (1E21)"])

    param = EPOCH_PARAMS

    l_budgets = [
        loss(n * 1e9, d * 1e12, param)
        for n, d in zip(
            df["N"],
            df["D"],
            strict=False,
        )
    ]
    n_opt, d_opt = zip(*[opt_params(l_budget, param) for l_budget in l_budgets], strict=False)
    n_opt = np.array(n_opt)
    d_opt = np.array(d_opt)
    df["N_opt"] = n_opt / 1e9
    df["D_opt"] = d_opt / 1e12
    df["FLOP_opt"] = 6 * df["N_opt"] * df["D_opt"]
    df["log10 FLOP_opt"] = np.log10(df["FLOP_opt"])


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
        "Model Size (B)" as N, 
        "Pretraining Data Size (T)" as D, 
        hash("Model Family") as "family_idx"
    FROM base_llm_benchmark_eval
    JOIN family_release_dates ON base_llm_benchmark_eval."Model Family" = family_release_dates.family
    """
).df()
base_llm_benchmarks = ["MMLU", "ARC-C", "HellaSwag", "Winograd", "GSM8K", "XWinograd"]


augment_df_opt_flops(
    base_llm_benchmark_eval,
)

openllm_elo_merged = duckdb.read_csv("./data_models/meta/openllm_elo_merged.csv")
openllm_elo_merged = duckdb.sql(
    """
    SELECT
        "chatbot_arena_name",
        "arena_score" as Elo,
        "IFEval Raw",
        "BBH Raw",
        "MATH Lvl 5 Raw",
        "GPQA Raw",
        "MUSR Raw",
        "MMLU-PRO Raw",
        year(release_date) + (1/365)*dayofyear(release_date) as release_date,
        "N",
        "D",
    FROM openllm_elo_merged
    """
).df()
augment_df_opt_flops(openllm_elo_merged)
openllm_elo_benchmarks = [
    "IFEval Raw",
    "BBH Raw",
    "MATH Lvl 5 Raw",
    "GPQA Raw",
    "MUSR Raw",
    "MMLU-PRO Raw",
]


# drop NaNs
base_llm_benchmark_eval.dropna(inplace=True)
openllm_elo_merged.dropna(inplace=True)

benchmark_data = [
    ("MMLU", 0.25),
    ("ARC-C", 0.2),
    ("HellaSwag", 0.25),
    ("Winograd", 0.5),
    ("TruthfulQA", 0.5),
    ("GSM8K", 0.0),
    ("XWinograd", 0.5),
    ("HumanEval", 0.0),
    ("IFEval Raw", 0.0),
    ("BBH Raw", 0.25),
    ("MATH Lvl 5 Raw", 0.0),
    ("GPQA Raw", 0.25),
    ("MUSR Raw", 0.3),
    ("MMLU-PRO Raw", 0.1),
]
benchmark_floor_dict = defaultdict(lambda: 0.0, {b: f for b, f in benchmark_data})


pc1model = LinearPC1Predictor(
    benchmarks=openllm_elo_benchmarks,
    benchmark_floors=[benchmark_floor_dict[b] for b in openllm_elo_benchmarks],
    train_model_scores=torch.tensor(openllm_elo_merged[openllm_elo_benchmarks].values),
)

openllm_elo_merged["PC1"] = pc1model.predict_capability_scores_from_model_scores(torch.tensor(openllm_elo_merged[openllm_elo_benchmarks].values))


# %%

# Create best fit line
z = np.polyfit(openllm_elo_merged["PC1"], openllm_elo_merged["Elo"], 1)

xspace = np.linspace(openllm_elo_merged["PC1"].min(), openllm_elo_merged["PC1"].max(), 100)
p = np.poly1d(z)
yline = p(xspace)


# Create the plot
plt.figure(figsize=(10,6))
scatter = plt.scatter(openllm_elo_merged["PC1"], openllm_elo_merged["Elo"], alpha=0.5, color="C0")
line = plt.plot(xspace, yline, color="red")

# Create equation string
slope, intercept = z
eq = f'y = {slope:.2f}x + {intercept:.2f}'

# Add legend
plt.legend([scatter, line[0]], ['Open Source Models', eq])

# Calculate correlation coefficient and R-squared
corr = np.corrcoef(openllm_elo_merged["PC1"], openllm_elo_merged["Elo"])[0,1]
r_squared = corr**2

# Update text annotation
plt.text(0.05, 0.92, f'R² = {r_squared:.2f}', 
         transform=plt.gca().transAxes, 
         fontsize=12)

# Add labels and title
plt.xlabel("PC-1 Score")
plt.ylabel("Elo Rating")
plt.title("Correlation between PC-1 Score and Elo Rating")


plt.grid(True, alpha=0.3)