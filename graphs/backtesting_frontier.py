# %%
from pathlib import Path
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import importlib
from importlib import reload
import duckdb
import matplotlib.axes
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors
import matplotlib.cm
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.special import expit
from pydantic import BaseModel
from sklearn.metrics import r2_score
import torch
import torch._dynamo.cache_size
import torch.nn.functional as F
import util_frontier
from util_frontier import (
    Frontier,
    get_running_top_n,
    get_running_top_n_2d,
    vectorized_highest_score_df,
    FrontierDatePredictor,
    FrontierFlopPredictor,
    FrontierDateToEloPredictor,
    FrontierDateBestElicitedToEloPredictor,
    FrontierDateAllElicitedToEloPredictor,
    FrontierDateDistinctElicitedToEloPredictor,
    FrontierDateToPC1Predictor,
    FrontierFlopDateToPC1Predictor,
    FrontierFlopDateToEloPredictor,
    FrontierFlopToEloPredictor,
    FrontierFlopToPC1Predictor,
)
from util_timeseries_backtesting import (
    BacktestSplitter,
    ExpandingWindowBacktestSplitter,
)
from util_obs_scaling_law_predictor import ScalingLaw
from util_linear_obs_scaling_law_predictor import LinearPC1Predictor


torch.set_num_threads(1)
torch._dynamo.cache_size.config.cache_size_limit = 1e9


matplotlib.rcParams["figure.dpi"] = 500


# Define the Chinchilla loss function parameter set
@dataclass
class ChinchillaParams:
    alpha: float
    beta: float
    A: float
    B: float
    E: float


# These numbers are from Epoch (Besiroglu et al. 2024)
EPOCH_PARAMS = ChinchillaParams(
    alpha=0.3478, beta=0.3658, A=482.01, B=2085.43, E=1.8172
)


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
    n_opt, d_opt = zip(
        *[opt_params(l_budget, param) for l_budget in l_budgets], strict=False
    )
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
        "Model" as model,
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

openllm_elo_merged_raw = duckdb.read_csv("./data_models/meta/openllm_elo_merged.csv")
openllm_elo_merged_raw = duckdb.sql(
    """
    SELECT
        "chatbot_arena_name" as model,
        "arena_score" as Elo,
        "IFEval Raw" as IFEval,
        "BBH Raw" as BBH,
        "MATH Lvl 5 Raw" as "MATH Lvl 5",
        "GPQA Raw" as GPQA,
        "MUSR Raw" as MUSR,
        "MMLU-PRO Raw" as "MMLU-PRO",
        year(release_date) + (1/365)*dayofyear(release_date) as release_date,
        "N",
        "D",
    FROM openllm_elo_merged_raw
    """
).df()
augment_df_opt_flops(openllm_elo_merged_raw)
openllm_elo_benchmarks = [
    "IFEval",
    "BBH",
    "MATH Lvl 5",
    "GPQA",
    "MUSR",
    "MMLU-PRO",
]


agentic_benchmark = duckdb.read_csv("./data_models/cache_new/agentic_benchmark.csv")
agentic_benchmark = duckdb.sql(
    """
    SELECT
        "Model" as model,
        "Chatbot Arena Elo" as Elo,
        "SWE-Bench Verified",
        "SWE-Bench Verified Elicited",
        "Cybench",
        "Cybench Elicited",
        "RE-Bench",
        "RE-Bench Elicited",
        year("Release Date") + (1/365)*dayofyear("Release Date") as release_date
    FROM agentic_benchmark
    UNION ALL VALUES
        ('o3', null, null, 0.715, null, null, null, null, 2025.00),
        ('claude-3.7-sonnet', null, null, 0.70, null, null, null, null, 2025.1507),
        ('gpt-4.5-preview',  null, null, 0.38, null, null, null, null, 2025.159)
    """
    # UNION ALL VALUES
    # ('a', 0, 0, 0, 0, 2025.00),
    # ('b', 0, 0, 0, 0, 2025.25),
    # ('c', 0, 0, 0, 0, 2025.50),
    # ('d', 0, 0, 0, 0, 2025.75),
    # ('e', 0, 0, 0, 0, 2026.00)
).df()
agentic_benchmark_benchmarks = ["SWE-Bench Verified", "Cybench", "RE-Bench"]


swebench_scaffolds = duckdb.read_csv("./data_models/cache_new/swebench_scaffolds.csv")
swebench_scaffolds = duckdb.sql(
    """
    SELECT
        "Model" as model,
        "SWE-Bench Verified",
        year("Release Date") + (1/365)*dayofyear("Release Date") as release_date
    FROM swebench_scaffolds
    """
).df()


cybench_scaffolds = duckdb.read_csv("./data_models/cache_new/cybench_scaffolds.csv")
cybench_scaffolds = duckdb.sql(
    """
    SELECT
        "Model" as model,
        "Cybench",
        year("Release Date") + (1/365)*dayofyear("Release Date") as release_date
    FROM cybench_scaffolds
    """
).df()


# drop NaNs
base_llm_benchmark_eval.dropna(inplace=True)
openllm_elo_merged = openllm_elo_merged_raw.dropna()

benchmark_data = [
    ("MMLU", 0.25, None),
    ("ARC-C", 0.2, None),
    ("HellaSwag", 0.25, None),
    ("Winograd", 0.5, None),
    ("TruthfulQA", 0.5, None),
    ("GSM8K", 0.0, None),
    ("XWinograd", 0.5, None),
    ("HumanEval", 0.0, None),
    ("IFEval", 0.0, None),
    ("BBH", 0.25, None),
    ("MATH Lvl 5", 0.0, None),
    ("GPQA", 0.25, None),
    ("MUSR", 0.3, None),
    ("MMLU-PRO", 0.1, None),
    ("SWE-Bench Verified", 0, 1),
    ("Cybench", 0, 1),
    ("RE-Bench", 0, 1.665714286),
]
benchmark_floor_dict = defaultdict(lambda: 0.0, {b: f for b, f, _ in benchmark_data})
benchmark_ceiling_dict = defaultdict(lambda: None, {b: c for b, _, c in benchmark_data})


@dataclass
class Spe:
    """
    Scatter Plot Entry
    """

    y_key: str
    y_label: str
    color: str
    alpha: float = 0.5


def plot_train_test(
    ax: matplotlib.axes.Axes,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    x_key: str,
    entries: list[Spe],
    title: str | None = None,
    y_label: str | None = None,
):
    for e in entries:
        ax.scatter(
            train_df[x_key],
            train_df[e.y_key],
            marker="x",
            alpha=e.alpha,
            color=e.color,
        )
        ax.scatter(
            test_df[x_key],
            test_df[e.y_key],
            marker="o",
            label=e.y_label,
            alpha=e.alpha,
            color=e.color,
        )
    ax.legend()

    ax.set_xlabel(x_key)
    if y_label is not None:
        ax.set_ylabel(y_label)
    if title is None and y_label is not None:
        title = f"{y_label} vs {x_key}"
    if title is not None:
        ax.set_title(title)


def year_formatter(x: float, _):
    year = int(x)
    month = int((x % 1) * 12 + 1)
    return f"{year}-{month:02d}"


def augment_df_slaw(model: Frontier, slaw: ScalingLaw, df_to_augment: pd.DataFrame):
    model_scores = torch.tensor(
        df_to_augment[model.necessary_benchmarks()].values, dtype=torch.float32
    )
    df_to_augment[f"{slaw.benchmark} pred"] = (
        slaw.forward(model.predict_frontier_capability_scores(model_scores))
        .detach()
        .numpy()
    )
    df_to_augment[f"{slaw.benchmark} capability score"] = df_to_augment[
        model.capability()
    ]
    df_to_augment[f"{slaw.benchmark} pred capability score"] = (
        model.predict_frontier_capability_scores(model_scores).detach().numpy()
    )


def augment_train_test_slaw(
    model: Frontier,
    slaw: ScalingLaw,
    train: pd.DataFrame,
    test: pd.DataFrame,
):
    augment_df_slaw(model, slaw, train)
    augment_df_slaw(model, slaw, test)


def augment_train_test_pc1(
    dataframe_benchmarks: list[str],
    predicted_benchmark: str,
    train: pd.DataFrame,
    test: Optional[pd.DataFrame],
) -> tuple[LinearPC1Predictor, pd.DataFrame, Optional[pd.DataFrame]]:
    # fit linear PC1 predictor on all models on all benchmarks except the predicted benchmark
    pc1_benchmarks = [b for b in dataframe_benchmarks if b != predicted_benchmark]
    pc1_benchmark_floors = [benchmark_floor_dict[b] for b in pc1_benchmarks]
    train_model_scores = torch.tensor(train[pc1_benchmarks].values, dtype=torch.float32)
    linpc1 = LinearPC1Predictor(
        benchmarks=pc1_benchmarks,
        benchmark_floors=pc1_benchmark_floors,
        train_model_scores=train_model_scores,
    )
    linpc1.fit()

    # augment the train and test data with the PC1 scores
    train = train.assign(
        PC1=(
            linpc1.predict_capability_scores_from_model_scores(train_model_scores)
            .detach()
            .numpy()
        )
    )
    if test is not None:
        test = test.assign(
            PC1=(
                linpc1.predict_capability_scores_from_model_scores(
                    torch.tensor(test[pc1_benchmarks].values, dtype=torch.float32)
                )
                .detach()
                .numpy()
            )
        )
    return linpc1, train, test


# filters benchmarks to exclude elicited values, which are generated on the fly
def raw_benchmarks(l: list[str]) -> list[str]:
    return [b for b in l if "elicited" not in b]


def get_training_data(
    ModelCls: type[Frontier],
    train: pd.DataFrame,
    fit_on_line: bool,
    slaw: ScalingLaw,
    predicted_benchmark: str,
    elicitation: Optional[pd.DataFrame],
):
    # create frontier train
    # TODO: support 2d benchmarks
    frontier_train_raw = train.dropna(
        subset=raw_benchmarks(ModelCls.necessary_benchmarks()) + [ModelCls.capability()]
    )
    if fit_on_line:
        x_col = ModelCls.necessary_benchmarks()[0]
        frontier_train = vectorized_highest_score_df(
            frontier_train_raw,
            x_col,
            np.linspace(
                frontier_train_raw[x_col].min(),
                frontier_train_raw[x_col].max(),
                100,
            ),
            ModelCls.capability(),
        )

    else:
        frontier_train = get_running_top_n(
            frontier_train_raw,
            ModelCls.necessary_benchmarks()[0],
            ModelCls.capability(),
            1,
            "model",
        )

    used_synthetic = False

    # elicited all basically adds in the elicitation of every frontier model, without thinking about selecting the best ones
    if ModelCls.uses_benchmark("elicited_all_frontier"):
        # sanity check
        assert not used_synthetic
        used_synthetic = True

        # set to 0 for all
        frontier_train["elicited_all_frontier"] = 0.0
        # compute the implied capability for each model
        frontier_train["implied_capability"] = (
            slaw.infer_capability_score_from_benchmark_score(
                torch.tensor(
                    frontier_train[f"{predicted_benchmark} Elicited"].values,
                    dtype=torch.float32,
                )
            )
            .detach()
            .numpy()
        )

        new_rows = []
        for row in frontier_train.to_dict(orient="records"):
            if not np.isnan(row["implied_capability"]):
                new_rows.append(
                    {
                        **row,
                        "elicited_all_frontier": 1.0,
                        ModelCls.capability(): row["implied_capability"],
                    }
                )
        # append the new rows
        frontier_train = pd.concat([frontier_train, pd.DataFrame(new_rows)])

    # elicited bests takes the standpoint that the best elicitation is the best predictor of future performance
    # past models with poor elicitation only are there because no one bothers to develop them further
    # therefore, we take the best elicitation as the best predictor of future performance
    if ModelCls.uses_benchmark("elicited_best"):
        # sanity check
        assert not used_synthetic
        used_synthetic = True

        # set to 0 for all
        frontier_train = frontier_train.assign(elicited_best=0.0)
        # compute the implied capability for each model
        frontier_train = frontier_train.assign(
            implied_capability=(
                slaw.infer_capability_score_from_benchmark_score(
                    torch.tensor(
                        frontier_train[f"{predicted_benchmark} Elicited"].values,
                        dtype=torch.float32,
                    )
                )
                .detach()
                .numpy()
            )
        )

        # compute the set of elicitation boosts by comparing it with the actual capability metric
        diffs = np.array(frontier_train["implied_capability"].values) - np.array(
            frontier_train[ModelCls.capability()].values
        )
        diffs = diffs[~np.isnan(diffs)]
        diffs_max = np.max(diffs, initial=0.0)
        print(diffs_max)

        new_rows = []
        for row in frontier_train.to_dict(orient="records"):
            print("row", row["implied_capability"] - row[ModelCls.capability()])
            if not np.isnan(row["implied_capability"]) and (
                row["implied_capability"] - row[ModelCls.capability()] >= diffs_max
            ):
                new_rows.append(
                    {
                        **row,
                        "elicited_best": 1.0,
                        ModelCls.capability(): row["implied_capability"],
                    }
                )
        # append the new rows
        frontier_train = pd.concat([frontier_train, pd.DataFrame(new_rows)])

    # elicited distinct takes the view that seperate curves apply for elicited and non-elicited models
    # therefore, we include all non-elicited frontier models, and then the frontier of (combined non-elicited and elicited) models
    # some rows will appear more than once. We flag the rows belonging to the second category by setting elicited_distinct to 1
    if ModelCls.uses_benchmark("elicited_distinct"):
        # sanity check
        assert not used_synthetic
        used_synthetic = True

        if elicitation is not None:
            # first, attempt to infer the score of all the scaffolds from the slaw
            elicited_df = elicitation.copy()
            elicited_df[ModelCls.capability()] = (
                slaw.infer_capability_score_from_benchmark_score(
                    torch.tensor(
                        elicited_df[predicted_benchmark].values,
                        dtype=torch.float32,
                    )
                )
                .detach()
                .numpy()
            )

            # concat the non-elicited models
            all_df = pd.concat([frontier_train_raw, elicited_df])
            all_df = all_df.assign(elicited_distinct=1.0)

            # get running top n
            all_frontier = get_running_top_n(
                all_df,
                ModelCls.necessary_benchmarks()[0],
                ModelCls.capability(),
                1,
                "model",
            )

            # set to 0 for all in the non-elicited set
            frontier_train = frontier_train.assign(elicited_distinct=0.0)

            # concat dataframes
            frontier_train = pd.concat([frontier_train, all_frontier])

        else:
            # set to 0 for all
            frontier_train = frontier_train.assign(elicited_distinct=0.0)

    return frontier_train


@dataclass
class BacktestDataPoint[T: Frontier]:
    split_train: pd.DataFrame
    split_test: pd.DataFrame
    model: T
    slaw: ScalingLaw

    def copy(self):
        """
        Returns a copy of the data point.
        The dataframes are deep copied, and the model and slaw are shallow copied.
        """
        return BacktestDataPoint(
            self.split_train.copy(),
            self.split_test.copy(),
            self.model,
            self.slaw,
        )


@dataclass
class BacktestFrontierData:
    splitter: BacktestSplitter
    model_class: type[Frontier]
    benchmarks: list[str]
    splits: list[str]
    # 2D array of BacktestDataPoint on the splits x benchmarks
    results: npt.NDArray[np.object_]
    # 1D array of BacktestDataPoint on the benchmarks (using all points)
    global_split_results: npt.NDArray[np.object_]


def backtest_models_frontier(
    splitter: BacktestSplitter,
    ModelCls: type[Frontier],
    dataframe: pd.DataFrame,
    dataframe_benchmarks: list[str],
    dataframe_benchmark_elicitations: Optional[list[Optional[pd.DataFrame]]] = None,
    fit_on_line: bool = False,
) -> BacktestFrontierData:
    # create object ndarray

    if dataframe_benchmark_elicitations is not None:
        assert len(dataframe_benchmarks) == len(
            dataframe_benchmark_elicitations
        ), f"{len(dataframe_benchmarks)} must equal {len(dataframe_benchmark_elicitations)}"
    else:
        dataframe_benchmark_elicitations = [None for _ in dataframe_benchmarks]

    train_test_splits = list(splitter.split(dataframe))

    data = BacktestFrontierData(
        splitter=splitter,
        model_class=ModelCls,
        benchmarks=dataframe_benchmarks,
        splits=[f"split_{i}" for i in range(len(train_test_splits))],
        results=np.empty(
            (len(train_test_splits), len(dataframe_benchmarks)), dtype=np.object_
        ),
        global_split_results=np.empty(len(dataframe_benchmarks), dtype=np.object_),
    )

    n_trains = (len(train_test_splits) + 1) * len(dataframe_benchmarks)

    for split_idx, (train, test) in enumerate(
        [(dataframe, dataframe.head(0))] + train_test_splits
    ):
        for bench_idx, predicted_benchmark in enumerate(dataframe_benchmarks):
            i_train = split_idx * len(dataframe_benchmarks) + bench_idx
            print(f"Training {i_train}/{n_trains}")

            if "PC1" in ModelCls.necessary_benchmarks() + [ModelCls.capability()]:
                _, train, test = augment_train_test_pc1(
                    dataframe_benchmarks,
                    predicted_benchmark,
                    train,
                    test,
                )
                assert test is not None
                assert "PC1" in train.columns

            # drop nan rows from the train set
            slaw_train = train.dropna(
                subset=[ModelCls.capability(), predicted_benchmark]
            )

            # train intermediate -> benchmark slaw
            slaw = ScalingLaw(
                benchmark=predicted_benchmark,
                floor=benchmark_floor_dict[predicted_benchmark],
                maybe_ceil=benchmark_ceiling_dict[predicted_benchmark],
                capability_scores=torch.tensor(
                    slaw_train[ModelCls.capability()].values, dtype=torch.float32
                ),
                benchmark_scores=torch.tensor(
                    slaw_train[predicted_benchmark].values, dtype=torch.float32
                ),
            )

            t0 = time.time()
            slaw.fit()
            print(
                f"{ModelCls.__name__} Scaling Law Training Time: {time.time() - t0:.2f} seconds"
            )

            frontier_train = get_training_data(
                ModelCls,
                train,
                fit_on_line,
                slaw,
                predicted_benchmark,
                dataframe_benchmark_elicitations[bench_idx],
            )

            # create model
            t0 = time.time()
            model = ModelCls(
                benchmark_data=torch.tensor(
                    frontier_train[ModelCls.necessary_benchmarks()].values,
                    dtype=torch.float32,
                ),
                capability_data=torch.tensor(
                    frontier_train[ModelCls.capability()].values, dtype=torch.float32
                ),
            )
            print(f"{ModelCls.__name__} Training Time: {time.time() - t0:.2f} seconds")

            # store the results
            if split_idx == 0:
                data.global_split_results[bench_idx] = BacktestDataPoint(
                    split_train=train,
                    split_test=test,
                    model=model,
                    slaw=slaw,
                )
            else:
                data.results[split_idx - 1, bench_idx] = BacktestDataPoint(
                    split_train=train,
                    split_test=test,
                    model=model,
                    slaw=slaw,
                )

    return data


def compute_test_train_error_frontier(
    arr: npt.NDArray[np.object_],
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
]:
    train_err = np.zeros_like(arr, dtype=np.float32)
    test_err = np.zeros_like(arr, dtype=np.float32)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            bdp: BacktestDataPoint[Frontier] = arr[i, j]
            train = bdp.split_train
            test = bdp.split_test
            model = bdp.model
            slaw = bdp.slaw

            for dataset, dataset_err_arr in ((train, train_err), (test, test_err)):
                dataset_frontier = get_running_top_n(
                    dataset, "release_date", slaw.benchmark, 1, "model"
                )

                x = torch.tensor(
                    dataset_frontier[model.necessary_benchmarks()].values,
                    dtype=torch.float32,
                )
                y = torch.tensor(
                    dataset_frontier[slaw.benchmark].values, dtype=torch.float32
                )
                y_hat = slaw.forward(model.predict_frontier_capability_scores(x))
                dataset_err_arr[i, j] = F.mse_loss(
                    y,
                    y_hat,
                ).item()

    return train_err, test_err


def compute_test_train_error_frontier_avg(
    arr: list[npt.NDArray[np.object_]],
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
]:
    train_err = np.zeros_like(arr[0], dtype=np.float32)
    test_err = np.zeros_like(arr[0], dtype=np.float32)
    for i in range(arr[0].shape[0]):
        for j in range(arr[0].shape[1]):
            bdp0: BacktestDataPoint[Frontier] = arr[0][i, j]
            train = bdp0.split_train
            test = bdp0.split_test
            benchmark = bdp0.slaw.benchmark

            for dataset, dataset_err_arr in ((train, train_err), (test, test_err)):
                dataset_frontier = get_running_top_n(
                    dataset, "release_date", benchmark, 1, "model"
                )
                
                y_hats = []
                for bpd in [bdp[i, j] for bdp in arr]:
                    model = bpd.model
                    slaw = bpd.slaw                   
                    
                    x = torch.tensor(
                        dataset_frontier[model.necessary_benchmarks()].values,
                        dtype=torch.float32,
                    )
                    y_hat = slaw.forward(model.predict_frontier_capability_scores(x))
                    y_hats.append(y_hat)
                
                y_hat_avg = torch.stack(y_hats).mean(dim=0)

                y = torch.tensor(
                    dataset_frontier[benchmark].values, dtype=torch.float32
                )
                dataset_err_arr[i, j] = F.mse_loss(
                    y,
                    y_hat_avg,
                ).item()

    return train_err, test_err



def plot_comparison(backtests: list[BacktestFrontierData], expand=False):
    assert len(backtests) > 0
    b0 = backtests[0]
    n_split, n_bench = b0.results.shape

    # key on which we split
    x_key = b0.splitter.key

    if expand:
        fig, ax = plt.subplots(
            n_split * len(backtests),
            n_bench,
            figsize=(4 * n_bench, 4 * n_split * len(backtests)),
            squeeze=False,
        )
    else:
        fig, ax = plt.subplots(
            n_split,
            n_bench,
            figsize=(4 * n_bench, 4 * n_split),
            squeeze=False,
        )

    for i, b in enumerate(backtests):
        # plot ground truth data
        for split_idx in range(n_split):
            if expand:
                y_idx = split_idx * len(backtests) + i
            else:
                y_idx = split_idx
            for bench_idx in range(n_bench):
                if i == 0 or expand:
                    b0dp: BacktestDataPoint[Frontier] = b0.results[split_idx, bench_idx]
                    # We plot the ground truth data
                    plot_train_test(
                        ax[y_idx, bench_idx],
                        b0dp.split_train,
                        b0dp.split_test,
                        x_key,
                        [Spe(b0dp.slaw.benchmark, "Ground Truth", "black")],
                        y_label=b0dp.slaw.benchmark,
                    )

                if expand:
                    color = "C0"
                else:
                    color = f"C{i}"

                # otherwise plot the model data
                bdp: BacktestDataPoint[Frontier] = b.results[split_idx, bench_idx]
                bdp_copy = bdp.copy()
                augment_train_test_slaw(
                    bdp_copy.model,
                    bdp_copy.slaw,
                    bdp_copy.split_train,
                    bdp_copy.split_test,
                )
                plot_train_test(
                    ax[y_idx, bench_idx],
                    bdp_copy.split_train,
                    bdp_copy.split_test,
                    x_key,
                    [
                        Spe(
                            f"{bdp_copy.slaw.benchmark} pred",
                            f"{type(bdp_copy.model).__name__} pred",
                            color,
                        ),
                    ],
                    y_label=bdp_copy.slaw.benchmark,
                )

    fig.tight_layout()
    plt.show()


def plot_errmatrix_comparison(
    backtests: list[BacktestFrontierData],
):
    assert len(backtests) > 0
    methods = [b.model_class.__name__.replace("Predictor", "") for b in backtests]
    # create 3 graphs for each split in [test, train]:
    # 1. Aggregate over benchmarks
    # 2. Aggregate over splits
    # 3. Aggregate over both
    fig, ax = plt.subplots(2, 3, figsize=(30 * 0.7, 20 * 0.7))

    # create 3d matrix of errors
    train_errs = np.zeros(
        (
            # methods
            len(backtests),
            # splits
            len(backtests[0].splits),
            # benchmarks
            len(backtests[0].benchmarks),
        )
    )
    test_errs = np.zeros_like(train_errs)  # same shape as err_train

    for i, b in enumerate(backtests):
        train_err, test_err = compute_test_train_error_frontier(b.results)
        train_errs[i] = train_err
        test_errs[i] = test_err

    train_vmax = np.max(np.sqrt(train_errs)).item()
    test_vmax = np.max(np.sqrt(test_errs)).item()

    # aggregate over splits
    sns.heatmap(
        np.sqrt(train_errs.mean(axis=1).T),
        ax=ax[0, 0],
        yticklabels=backtests[0].benchmarks,
        xticklabels=methods,
        annot=True,
        vmin=0,
        vmax=train_vmax,
    )
    sns.heatmap(
        np.sqrt(test_errs.mean(axis=1).T),
        ax=ax[1, 0],
        yticklabels=backtests[0].benchmarks,
        xticklabels=methods,
        annot=True,
        vmin=0,
        vmax=test_vmax,
    )

    # aggregate over benchmarks
    sns.heatmap(
        np.sqrt(train_errs.mean(axis=2).T),
        ax=ax[0, 1],
        yticklabels=backtests[0].splits,
        xticklabels=methods,
        annot=True,
        vmin=0,
        vmax=train_vmax,
    )
    sns.heatmap(
        np.sqrt(test_errs.mean(axis=2).T),
        ax=ax[1, 1],
        yticklabels=backtests[0].splits,
        xticklabels=methods,
        annot=True,
        vmin=0,
        vmax=test_vmax,
    )

    # aggregate over methods
    sns.heatmap(
        np.sqrt(train_errs.mean(axis=0).T),
        ax=ax[0, 2],
        yticklabels=backtests[0].benchmarks,
        xticklabels=backtests[0].splits,
        annot=True,
        vmin=0,
        vmax=train_vmax,
    )
    sns.heatmap(
        np.sqrt(test_errs.mean(axis=0).T),
        ax=ax[1, 2],
        yticklabels=backtests[0].benchmarks,
        xticklabels=backtests[0].splits,
        vmin=0,
        vmax=test_vmax,
        annot=True,
    )

    # set column titles
    ax[0, 0].set_title("Predictor perf on Benchmark", size="xx-large")
    ax[0, 1].set_title("Predictor perf on Split", size="xx-large")
    ax[0, 2].set_title("Overall perf on (Split, Benchmark)", size="xx-large")

    # set row titles
    ax[0, 0].set_ylabel("Train Set", size="xx-large")
    ax[1, 0].set_ylabel("Test Set", size="xx-large")

    fig.tight_layout()
    plt.show()


def plot_test_errmatrix_single(
    backtest: BacktestFrontierData,
):
    method = backtest.model_class.__name__.replace("Predictor", "")

    # Create a heatmap with the following rows

    # create 3 graphs for each split in [test, train]:
    # 1. Aggregate over benchmarks
    # 2. Aggregate over splits
    # 3. Aggregate over both
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), squeeze=False)

    _, test_err = compute_test_train_error_frontier(backtest.results)
    # make the last column of test_err a sum column
    test_err = np.concatenate([test_err, test_err.mean(axis=1, keepdims=True)], axis=1)
    # make the last row of test_err a sum row
    test_err = np.concatenate([test_err, test_err.mean(axis=0, keepdims=True)], axis=0)

    test_vmax = np.max(np.sqrt(test_err)).item()

    splits = []
    for result in backtest.results.T[0]:
        min_v = result.split_train[backtest.splitter.key].min()
        max_v = result.split_train[backtest.splitter.key].max()
        splits.append(f"{min_v:.1f} - {max_v:.1f}")

    benchmarks = backtest.benchmarks

    # aggregate over splits
    g = sns.heatmap(
        np.sqrt(test_err),
        ax=ax[0, 0],
        xticklabels=benchmarks + ["MEAN"],
        yticklabels=splits + ["MEAN"],
        annot=True,
        vmin=0,
        vmax=test_vmax,
    )
    g.set_yticklabels(
        g.get_yticklabels(),
        rotation=0,
    )

    ax[0, 0].hlines(
        len(backtest.splits), *ax[0, 0].get_xlim(), color="black", linewidth=4
    )
    ax[0, 0].vlines(
        len(backtest.benchmarks), *ax[0, 0].get_ylim(), color="black", linewidth=4
    )
    ax[0, 0].set_xlabel("Benchmarks", size="large")

    # set column titles
    ax[0, 0].set_title(f"{method} perf on Benchmark", size="x-large")

    fig.tight_layout()
    plt.show()


def plot_all_loss_curves(data: BacktestFrontierData):
    n_split, n_bench = data.results.shape
    fig, ax = plt.subplots(
        n_split + 1,
        n_bench,
        figsize=(4 * n_bench, 4 * (n_split + 1)),
        squeeze=False,
    )
    for split_idx in range(n_split + 1):
        for bench_idx in range(n_bench):
            bdp: BacktestDataPoint[Frontier] = (
                data.results[split_idx, bench_idx]
                if split_idx < n_split
                else data.global_split_results[bench_idx]
            )
            slaw = bdp.slaw
            ax[split_idx, bench_idx].plot(np.log10(slaw.train_losses[:]), label="train")
            ax[split_idx, bench_idx].set_title(slaw.benchmark)
            ax[split_idx, bench_idx].legend()
    plt.show()


def plot_split(
    backtest: BacktestFrontierData,
    benchmark_id: int,
    x_key: str,
    expand=False,
    line=False,
    capability=False,
):
    color_list = [
        "tab:blue",
        "tab:cyan",
        "tab:green",
        "tab:orange",
    ]

    n_split, n_bench = backtest.results.shape
    assert benchmark_id < n_bench

    assert not (
        line and x_key != backtest.splitter.key
    ), "Cannot plot line without x_key being the split key"

    if expand:
        fig, ax = plt.subplots(1, n_split, figsize=(5 * n_split, 5), squeeze=False)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), squeeze=False)

    # first, plot the train points.
    # We need to use the final dataframe to get the ground truth data, since it will have the best-trained PC-1 score (and doesn't matter for the other models)
    bdp_g: BacktestDataPoint[Frontier] = backtest.global_split_results[benchmark_id]
    bdp_g_copy = bdp_g.copy()
    print(bdp_g_copy.model)

    augment_df_slaw(
        bdp_g_copy.model,
        bdp_g_copy.slaw,
        bdp_g_copy.split_train,
    )

    bdp_g_splits = list(backtest.splitter.split(bdp_g_copy.split_train))

    frontier_set = set(
        get_running_top_n(
            bdp_g_copy.split_train,
            backtest.splitter.key,
            bdp_g.slaw.benchmark,
            1,
            "model",
        )["model"]
    )

    last_max_v = bdp_g_copy.split_train[backtest.splitter.key].min()

    for j in range(len(bdp_g_splits) if expand else 1):
        curr_ax = ax[0, j]

        plotted_points = set()

        for i, (train, test) in enumerate(bdp_g_splits):
            max_v = train[backtest.splitter.key].max()
            min_v = last_max_v
            last_max_v = max_v

            df = train[~train[backtest.splitter.key].isin(plotted_points)]

            if capability:
                y_key = f"{bdp_g.slaw.benchmark} capability score"
            else:
                y_key = f"{bdp_g.slaw.benchmark}"

            curr_ax.scatter(
                df[x_key],
                df[y_key],
                label=f"{min_v:.1f} - {max_v:.1f} {backtest.splitter.key}",
                alpha=[1 if m in frontier_set else 0.75 for m in df["model"]],
                s=[40 if m in frontier_set else 20 for m in df["model"]],
                color=color_list[i],
            )
            curr_ax.set_title(f"{x_key} vs {y_key}")
            curr_ax.set_xlabel(x_key)
            curr_ax.set_ylabel(bdp_g.slaw.benchmark)

            plotted_points.update(train[backtest.splitter.key])

            if i == len(bdp_g_splits) - 1:
                df = bdp_g_copy.split_train[
                    ~bdp_g_copy.split_train[backtest.splitter.key].isin(plotted_points)
                ]
                curr_ax.scatter(
                    df[x_key],
                    df[y_key],
                    label=f"{max_v:.1f} + {backtest.splitter.key}",
                    alpha=[1 if m in frontier_set else 0.75 for m in df["model"]],
                    s=[40 if m in frontier_set else 20 for m in df["model"]],
                    color=color_list[len(bdp_g_splits)],
                )
                curr_ax.set_title(f"{x_key} vs {y_key}")
                curr_ax.set_xlabel(x_key)
                curr_ax.set_ylabel(y_key)

    # now plot the predictions
    # to do this, we use the model to make predictions for the entire space and plot it

    for split_idx in range(n_split):
        color = color_list[split_idx]
        bdp: BacktestDataPoint[Frontier] = backtest.results[split_idx, benchmark_id]

        # augment the global split with the model's predictions
        bdp_g_copy2 = bdp_g.copy()
        augment_df_slaw(bdp.model, bdp.slaw, bdp_g_copy2.split_train)

        if expand:
            curr_ax = ax[0, split_idx]
        else:
            curr_ax = ax[0, 0]

        if capability:
            label = f"{type(bdp.model).__name__} capability"
            y_key = f"{bdp.slaw.benchmark} pred capability score"
        else:
            label = f"{type(bdp.model).__name__} pred"
            y_key = f"{bdp.slaw.benchmark} pred"

        # plot the predictions
        if line:
            xs = np.array(bdp_g_copy.split_train[x_key])
            ys = np.array(bdp_g_copy2.split_train[y_key])

            # Sort both arrays based on x values
            sort_idx = np.argsort(xs)

            curr_ax.plot(
                xs[sort_idx],
                ys[sort_idx],
                label=label,
                alpha=1,
                color=color,
            )
        else:
            curr_ax.scatter(
                bdp_g_copy.split_train[x_key],
                bdp_g_copy2.split_train[y_key],
                label=label,
                alpha=1,
                marker="x",
                color=color,
            )

        min_v = bdp_g_copy.split_train[backtest.splitter.key].min()
        max_v = bdp_g_copy.split_train[backtest.splitter.key].max()

        curr_ax.legend()

    fig.tight_layout()
    plt.show()


def compute_density_samples_via_bootstrapping(
    ModelCls: type[Frontier],
    train: pd.DataFrame,
    x_linspace: npt.NDArray[np.float32],
    target_benchmark: str,
    canonical_slaw: ScalingLaw,
    extra_data: Optional[pd.DataFrame],
    n_samples: int = 1000,
    percentage_drawn: float = 1,
    fit_on_line: bool = False,
    jitter: float = 1.0,
    elicit=False,
) -> npt.NDArray[np.float32]:
    # this method works by randomly dropping frontier models and training a bunch of entries in parallel

    n_bootstrap = n_samples

    frontier_train_df = get_training_data(
        ModelCls,
        train,
        fit_on_line=fit_on_line,
        slaw=canonical_slaw,
        predicted_benchmark=target_benchmark,
        elicitation=extra_data,
    )

    frontier_train_tensor = torch.tensor(
        frontier_train_df[
            ModelCls.necessary_benchmarks() + [ModelCls.capability()]
        ].values,
        dtype=torch.float32,
    )

    n_samples_per_bootstrap_frontier = int(
        percentage_drawn * frontier_train_tensor.shape[0]
    )

    slaw_train_tensor = torch.tensor(
        train.dropna(subset=["Elo", target_benchmark])[
            [
                "Elo",
                target_benchmark,
            ]
        ].values,
        dtype=torch.float32,
    )

    n_samples_per_bootstrap_slaw = int(percentage_drawn * slaw_train_tensor.shape[0])

    # shape: (n_bootstrap, n_samples_per_bootstrap_frontier, 3)
    frontier_train_tensor_resampled = torch.index_select(
        input=frontier_train_tensor,
        dim=0,
        index=torch.randint(
            0,
            frontier_train_tensor.shape[0],
            (n_bootstrap * n_samples_per_bootstrap_frontier,),
        ),
    ).reshape(n_bootstrap, n_samples_per_bootstrap_frontier, -1)

    # shape: (n_bootstrap, n_samples_per_bootstrap, 2)
    slaw_train_tensor_resampled = torch.index_select(
        input=slaw_train_tensor,
        dim=0,
        index=torch.randint(
            0, slaw_train_tensor.shape[0], (n_bootstrap * n_samples_per_bootstrap_slaw,)
        ),
    ).reshape(n_bootstrap, n_samples_per_bootstrap_slaw, 2)

    # add jitter to the year of magnitude 0.01
    frontier_train_tensor_resampled[:, :, 0] += (
        torch.randn(n_bootstrap, n_samples_per_bootstrap_frontier) * 0.01 * jitter
    )

    # add jitter to elo of magnitude 1
    frontier_train_tensor_resampled[:, :, -1] += (
        torch.randn(n_bootstrap, n_samples_per_bootstrap_frontier) * jitter
    )
    slaw_train_tensor_resampled[:, :, 0] += (
        torch.randn(n_bootstrap, n_samples_per_bootstrap_slaw) * jitter
    )

    model = ModelCls(
        # benchmark data (n_bootstrap, n_samples_per_bootstrap_frontier, 2)
        frontier_train_tensor_resampled[:, :, 0:-1],
        # capability data (n_bootstrap, n_samples_per_bootstrap)
        frontier_train_tensor_resampled[:, :, -1],
    )

    # compute sigmoid relationship between elo and the target benchmark on all data
    slaw = ScalingLaw(
        benchmark=target_benchmark,
        floor=benchmark_floor_dict[target_benchmark],
        maybe_ceil=benchmark_ceiling_dict[target_benchmark],
        # elo
        capability_scores=slaw_train_tensor_resampled[:, :, 0],
        # target benchmark
        benchmark_scores=slaw_train_tensor_resampled[:, :, 1],
    )

    # now fit sigmoid for all, we can use slaw already since it is vectorized
    t0 = time.time()
    slaw.fit()
    print(f"Fit time: {time.time() - t0:.2f} seconds")

    # now we can do inference for all

    # first, compute the linear regression
    # shape: (n_bootstrap, n_linspace)
    capability_score = model.predict_frontier_capability_scores(
        torch.stack(
            [
                torch.tensor(x_linspace, dtype=torch.float32),
                torch.full_like(
                    torch.tensor(x_linspace, dtype=torch.float32),
                    fill_value=1.0 if elicit else 0.0,
                ),
            ],
            dim=-1,
        ).expand(n_bootstrap, len(x_linspace), 2)
    )

    # now pass it through the slaw
    # shape: (n_bootstrap, n_linspace)
    benchmark_score = slaw(capability_score)
    # benchmark_score = capability_score

    # to numpy and return
    # shape: (n_bootstrap, n_linspace)
    return benchmark_score.detach().numpy()


def compute_density_samples(
    train: pd.DataFrame,
    x_linspace: npt.NDArray[np.float32],
    target_benchmark: str,
    n_samples: int = 1000,
    elicitation_boost: float = 0.0,
) -> npt.NDArray[np.float32]:

    # Get benchmark floor and ceiling
    b_f = benchmark_floor_dict[target_benchmark]
    b_c = benchmark_ceiling_dict[target_benchmark]
    b_c = 1.0 if b_c is None else b_c

    # Drop rows with missing values in relevant columns
    train = train.dropna(subset=["Elo", "release_date", target_benchmark])

    # --- Linear Regression (A -> B) ---
    # Get frontier models (top Elo per release_date)
    frontier_train_df = get_running_top_n(train, "release_date", "Elo", 1, "model")

    # Fit linear regression (release_date -> Elo)
    X_linear = sm.add_constant(frontier_train_df["release_date"].values)
    y_linear = frontier_train_df["Elo"].values
    linear_model = sm.OLS(y_linear, X_linear).fit()
    linear_cov = linear_model.cov_params()

    # --- Logistic Regression (B -> C) ---
    # Use ALL training data with valid Elo and target_benchmark
    logistic_train = train[["Elo", target_benchmark]].dropna()
    X_logistic = sm.add_constant(logistic_train["Elo"].values)  # Add intercept
    y_logistic = (np.array(logistic_train[target_benchmark].values) - b_f) / (
        b_c - b_f
    )  # Normalize to [0, 1]
    logistic_model = sm.Logit(y_logistic, X_logistic).fit(disp=0)
    logistic_cov = logistic_model.cov_params()

    # --- Prediction Matrix ---
    X_pred = sm.add_constant(x_linspace)  # For linear model

    # --- Sampling ---
    samples = []
    for _ in range(n_samples):
        # Sample linear regression parameters
        beta_linear = np.random.multivariate_normal(linear_model.params, linear_cov)

        # Sample logistic regression parameters
        beta_logistic = np.random.multivariate_normal(
            logistic_model.params, logistic_cov
        )

        # Predict Elo scores (B)
        y_pred = X_pred @ beta_linear + elicitation_boost

        # Calculate logit (η) using sampled parameters
        eta = beta_logistic[0] + beta_logistic[1] * y_pred

        # Convert to probability (C)
        probs = (b_c - b_f) * expit(eta) + b_f

        samples.append(probs)

    return np.stack(samples)  # Shape: (n_samples, len(x_linspace))


@dataclass
class HistogramFigure:
    """
    Contains some extra data for helping to lay out the histogram figure in Results
    """

    benchmark_id: int
    y_level: float
    elicitations: Optional[pd.DataFrame] = None


def get_first_crossing(
    xs: npt.NDArray[np.float32], ys: npt.NDArray[np.float32], y: float
) -> Optional[float]:
    """
    Find the first crossing of the line y = y with the curve defined by xs and ys
    """
    for i in range(len(xs) - 1):
        if ys[i] < y and ys[i + 1] >= y:
            # interpolate
            x1, x2 = xs[i], xs[i + 1]
            y1, y2 = ys[i], ys[i + 1]
            return x1 + (x2 - x1) * (y - y1) / (y2 - y1)
    return None


class ForecastData(BaseModel):
    x_linspace: list[float]
    y_linspace: list[float]
    forecast: list[float]
    min_ci: list[float]
    max_ci: list[float]
    density: list[list[float]]


def compute_forecast_and_density(
    backtest: BacktestFrontierData,
    benchmark_id: int,
    x_linspace: npt.NDArray[np.float32],
    y_linspace: npt.NDArray[np.float32],
    elicitations: Optional[pd.DataFrame] = None,
) -> tuple[
    ForecastData,
    Optional[ForecastData],
]:
    n_samples = 10_000

    bdp_g: BacktestDataPoint[Frontier] = backtest.global_split_results[benchmark_id]
    bdp_g = bdp_g.copy()
    augment_bdp_elicited_best(bdp_g)

    do_elicit = (
        isinstance(bdp_g.model, FrontierDateAllElicitedToEloPredictor)
        or isinstance(bdp_g.model, FrontierDateBestElicitedToEloPredictor)
        or isinstance(bdp_g.model, FrontierDateDistinctElicitedToEloPredictor)
    ) and elicitations is not None

    x_key = bdp_g.slaw.benchmark

    samples = compute_density_samples_via_bootstrapping(
        type(bdp_g.model),
        bdp_g.split_train,
        x_linspace,
        x_key,
        canonical_slaw=bdp_g.slaw,
        extra_data=None,
        n_samples=n_samples,
        percentage_drawn=2 if do_elicit else 1,
        jitter=10,
    )

    min_ci = np.percentile(samples, 2.5, axis=0)
    max_ci = np.percentile(samples, 97.5, axis=0)

    xs_t = torch.tensor(x_linspace, dtype=torch.float32).reshape(-1, 1)

    forecast = (
        bdp_g.slaw.forward(
            bdp_g.model.predict_frontier_capability_scores(
                torch.cat([xs_t, torch.zeros_like(xs_t)], dim=1)
            )
        )
        .detach()
        .numpy()
    )
    if do_elicit:
        forecast_elicited = (
            bdp_g.slaw.forward(
                bdp_g.model.predict_frontier_capability_scores(
                    torch.cat([xs_t, torch.ones_like(xs_t)], dim=1)
                )
            )
            .detach()
            .numpy()
        )
        samples_elicited = compute_density_samples_via_bootstrapping(
            type(bdp_g.model),
            bdp_g.split_train,
            x_linspace,
            x_key,
            canonical_slaw=bdp_g.slaw,
            extra_data=elicitations,
            n_samples=n_samples,
            percentage_drawn=2,
            jitter=10,
            elicit=True,
        )
        min_ci_elicited = np.percentile(samples_elicited, 2.5, axis=0)
        max_ci_elicited = np.percentile(samples_elicited, 97.5, axis=0)
    else:
        forecast_elicited = None
        samples_elicited = None
        min_ci_elicited = None
        max_ci_elicited = None

    # Expand samples to include x coordinates
    x_coords = np.tile(x_linspace, (n_samples, 1))  # Shape: n_samples x n_linspace
    y_coords = samples  # Shape: n_samples x n_linspace

    # Flatten arrays for histogram2d
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()

    # Create density grid

    density_grid, _, _ = np.histogram2d(
        x_flat, y_flat, bins=(x_linspace, y_linspace)  # type: ignore
    )

    # Normalize
    density_grid: npt.NDArray[np.float32] = density_grid / (n_samples * len(x_linspace))

    # Do the same for elicited samples if they exist
    if samples_elicited is not None:
        y_coords_elicited = samples_elicited
        x_coords_elicited = np.tile(x_linspace, (n_samples, 1))

        count_grid_elicited, _, _ = np.histogram2d(
            x_coords_elicited.flatten(),
            y_coords_elicited.flatten(),
            bins=(x_linspace, y_linspace),  # type: ignore
        )
        density_grid_elicited: Optional[npt.NDArray[np.float32]] = (
            count_grid_elicited / (n_samples * len(x_linspace))
        )
    else:
        density_grid_elicited = None

    forecast = ForecastData(
        x_linspace=x_linspace.tolist(),
        y_linspace=y_linspace.tolist(),
        forecast=forecast.tolist(),
        min_ci=min_ci.tolist(),
        max_ci=max_ci.tolist(),
        density=density_grid.tolist(),
    )
    if (
        forecast_elicited is not None
        and min_ci_elicited is not None
        and max_ci_elicited is not None
        and density_grid_elicited is not None
    ):
        forecast_elicited = ForecastData(
            x_linspace=x_linspace.tolist(),
            y_linspace=y_linspace.tolist(),
            forecast=forecast_elicited.tolist(),
            min_ci=min_ci_elicited.tolist(),
            max_ci=max_ci_elicited.tolist(),
            density=density_grid_elicited.tolist(),
        )
    else:
        forecast_elicited = None

    return forecast, forecast_elicited


def plot_histogram_for_all(
    backtest: BacktestFrontierData,
    histograms: list[HistogramFigure],
    min_date: float = 2024,
    max_date: float = 2035,
):
    fig, axs = plt.subplots(
        len(histograms), 1, figsize=(12, 1.5 * len(histograms)), squeeze=False
    )

    n_samples = 10_000

    for i, hist in enumerate(histograms):
        bdp_g: BacktestDataPoint[Frontier] = backtest.global_split_results[
            hist.benchmark_id
        ]
        bdp_g = bdp_g.copy()
        augment_bdp_elicited_best(bdp_g)

        do_elicit = (
            isinstance(bdp_g.model, FrontierDateAllElicitedToEloPredictor)
            or isinstance(bdp_g.model, FrontierDateBestElicitedToEloPredictor)
            or isinstance(bdp_g.model, FrontierDateDistinctElicitedToEloPredictor)
        ) and hist.elicitations is not None

        x_key = bdp_g.slaw.benchmark
        x_linspace = np.linspace(min_date, max_date, 200)

        y_level = hist.y_level

        official_forecast = get_first_crossing(
            x_linspace,
            bdp_g.slaw.forward(
                bdp_g.model.predict_frontier_capability_scores(
                    torch.tensor(
                        np.stack([x_linspace, np.zeros_like(x_linspace)], axis=1),
                        dtype=torch.float32,
                    )
                )
            )
            .detach()
            .numpy(),
            y_level,
        )

        samples = compute_density_samples_via_bootstrapping(
            type(bdp_g.model),
            bdp_g.split_train,
            x_linspace,
            x_key,
            canonical_slaw=bdp_g.slaw,
            extra_data=None,
            n_samples=n_samples,
            percentage_drawn=2 if do_elicit else 1,
            jitter=10,
        )

        if do_elicit:
            samples_elicited = compute_density_samples_via_bootstrapping(
                type(bdp_g.model),
                bdp_g.split_train,
                x_linspace,
                x_key,
                canonical_slaw=bdp_g.slaw,
                extra_data=hist.elicitations,
                n_samples=n_samples,
                percentage_drawn=2,
                jitter=10,
                elicit=True,
            )
        else:
            samples_elicited = None

        ax = axs[i, 0]

        n_samples, n_linspace = samples.shape

        # count all samples within 0.01 of y_level
        samples_count_at_y_level = np.sum(np.abs(samples - y_level) < 0.02, axis=0)
        density = samples_count_at_y_level / n_samples

        # create a histogram manually
        ax.stairs(
            density[:-1], x_linspace, fill=True, alpha=0.4, color="tab:blue", zorder=0
        )

        unelicited_lines = []
        elicited_lines = []

        forecast_lw = 2.5

        # plot official forecast
        if official_forecast is not None:
            ax.axvline(
                x=official_forecast,
                color="tab:blue",
                lw=forecast_lw,
                linestyle="solid",
                label="Forecast" if i == 0 else None,
                zorder=1,
            )
            unelicited_lines.append(
                Line2D(
                    [0],
                    [0],
                    color="tab:blue",
                    lw=forecast_lw,
                    linestyle="solid",
                    label=year_formatter(official_forecast, None),
                )
            )
            # yscale = ax.get_ylim()[1] - ax.get_ylim()[0]
            # ax.annotate(
            #     xy=(official_forecast, yscale * 0.5),
            #     xytext=(official_forecast + 2.5, yscale * 0.5),
            #     size="small",
            #     text=f"{year_formatter(official_forecast, None)}",
            #     color="black",
            #     horizontalalignment="center",
            #     bbox={"facecolor": "black", "alpha": 0.3, "pad": 3},
            #     arrowprops=dict(facecolor="black", arrowstyle="->"),
            # )

        # Calculate cumulative density and find quantiles
        cumulative_density = np.cumsum(density) / np.sum(density)

        quantiles = [0.025, 0.5, 0.975]
        labels = ["2.5th", "50th", "97.5th"]
        linestyles = [(0, (1, 1)), "dashed", (0, (1, 3))]

        # Calculate quantile values using linear interpolation
        quantile_values = np.interp(quantiles, cumulative_density, x_linspace)

        for q_val, l, linestyle in zip(quantile_values, labels, linestyles):
            ax.axvline(
                x=q_val,
                color="tab:blue",
                linestyle=linestyle,
                label=f"{l} %ile" if i == 0 else None,
                zorder=1,
            )
            unelicited_lines.append(
                Line2D(
                    [0],
                    [0],
                    color="tab:blue",
                    linestyle=linestyle,
                    label=year_formatter(q_val, None),
                )
            )

            # yscale = ax.get_ylim()[1] - ax.get_ylim()[0]
            # ax.annotate(
            #     xy=(q_val, yscale * 0.5),
            #     xytext=(
            #         q_val + (0.5 if l == "50th" else 0.75),
            #         yscale * 0.1,
            #     ),
            #     size="small",
            #     text=f"{year_formatter(q_val, None)}",
            #     color="black",
            #     horizontalalignment="center",
            #     bbox={"facecolor": "tab:blue", "alpha": 0.3, "pad": 3},
            #     arrowprops=dict(facecolor="black", arrowstyle="->"),
            # )

        if samples_elicited is not None:

            official_forecast = get_first_crossing(
                x_linspace,
                bdp_g.slaw.forward(
                    bdp_g.model.predict_frontier_capability_scores(
                        torch.tensor(
                            np.stack([x_linspace, np.ones_like(x_linspace)], axis=1),
                            dtype=torch.float32,
                        )
                    )
                )
                .detach()
                .numpy(),
                y_level,
            )

            # count all samples within 0.01 of y_level
            samples_count_at_y_level = np.sum(
                np.abs(samples_elicited - y_level) < 0.02, axis=0
            )
            density = samples_count_at_y_level / n_samples

            # create a histogram manually
            ax.stairs(
                density[:-1],
                x_linspace,
                fill=True,
                alpha=0.4,
                color="tab:orange",
                zorder=2,
            )
            yscale = ax.get_ylim()[1] - ax.get_ylim()[0]

            # plot official forecast
            if official_forecast is not None:
                ax.axvline(
                    x=official_forecast,
                    color="tab:orange",
                    lw=forecast_lw,
                    linestyle="solid",
                    label="Forecast (elicited)" if i == 0 else None,
                    zorder=3,
                )
                elicited_lines.append(
                    Line2D(
                        [0],
                        [0],
                        color="tab:orange",
                        lw=forecast_lw,
                        linestyle="solid",
                        label=year_formatter(official_forecast, None),
                    )
                )

                # yscale = ax.get_ylim()[1] - ax.get_ylim()[0]
                # ax.annotate(
                #     xy=(official_forecast, yscale * 0.5),
                #     xytext=(official_forecast - 2, yscale * 0.5),
                #     size="small",
                #     text=f"{year_formatter(official_forecast, None)}",
                #     color="black",
                #     horizontalalignment="center",
                #     bbox={"facecolor": "black", "alpha": 0.3, "pad": 3},
                #     arrowprops=dict(facecolor="black", arrowstyle="->"),
                # )

            # Calculate cumulative density and find quantiles
            cumulative_density = np.cumsum(density) / np.sum(density)

            # Calculate quantile values using linear interpolation
            quantile_values = np.interp(quantiles, cumulative_density, x_linspace)

            for j, (q_val, l, linestyle) in enumerate(
                zip(quantile_values, labels, linestyles)
            ):
                ax.axvline(
                    x=q_val,
                    color="tab:orange",
                    linestyle=linestyle,
                    label=f"{l} %ile (elicited)" if i == 0 else None,
                    zorder=3,
                )

                elicited_lines.append(
                    Line2D(
                        [0],
                        [0],
                        color="tab:orange",
                        linestyle=linestyle,
                        label=year_formatter(q_val, None),
                    )
                )

                # disp = ((0.5 if l == "50th" else 0.75),)

                # ax.annotate(
                #     xy=(q_val, yscale * 0.5),
                #     xytext=(
                #         2024 + j,
                #         yscale * j * 0.2,
                #     ),
                #     size="small",
                #     text=f"{year_formatter(q_val, None)}",
                #     color="black",
                #     horizontalalignment="center",
                #     bbox={"facecolor": "tab:orange", "alpha": 0.3, "pad": 2},
                #     arrowprops=dict(facecolor="black", arrowstyle="->"),
                # )

        legend_lines = elicited_lines + unelicited_lines

        ax.set_title(f"{x_key} at {y_level}", size="medium")
        # set xticks at multiples of 2
        ax.set_xticks(np.arange(min_date, max_date + 1, 2))
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.set_yticks([])
        ax.legend(
            handles=legend_lines,
            loc="upper right",
            ncol=len(legend_lines) // 4,
            prop={"size": 8},
        )
    fig.subplots_adjust(hspace=0.5, right=0.82)
    fig.legend(loc="center right", ncol=1)
    return fig


@dataclass
class Figure1Benchmark:
    """
    Contains some extra data for helping to lay out Figure 1
    """

    id: int
    yaxis: str
    density: bool = False
    fit_on_line: bool = False
    elicitations: Optional[pd.DataFrame] = None


model_friendly_names = {
    "anthropic/claude-3-5-sonnet-20241022": "claude-3-5-sonnet",
    "openai/gpt-4o-2024-08-06": "gpt-4o",
    "openai/gpt-4o-mini-2024-07-18": "gpt-4o-mini",
    "openai/gpt-3.5-turbo-0125": "gpt-3.5-turbo",
    "openai/gpt-4-turbo-2024-04-09": "gpt-4-turbo",
    "openai/o1-2024-12-17": "o1",
    "openai/o1-mini-2024-09-12": "o1-mini",
    "anthropic/claude-3-5-sonnet-20240620": "claude-3-5-sonnet (old)",
    "anthropic/claude-3-opus-20240229": "claude-3-opus",
    "together/Qwen--Qwen2.5-72B-Instruct-Turbo": "qwen2.5-72b",
    "o3": "o3",
    "claude-3.7-sonnet": "claude-3.7-sonnet",
    "gpt-4.5-preview": "gpt-4.5-preview",
}


model_markers = {
    "anthropic/claude-3-5-sonnet-20241022": "v",
    "openai/gpt-4o-2024-08-06": "P",
    "openai/gpt-3.5-turbo-0125": "D",
    "openai/gpt-4-turbo-2024-04-09": "+",
    "openai/o1-2024-12-17": "x",
    "openai/o1-mini-2024-09-12": "d",
    "anthropic/claude-3-5-sonnet-20240620": "X",
    "anthropic/claude-3-opus-20240229": "s",
    "o3": "D",
    "claude-3.7-sonnet": "P",
    "gpt-4.5-preview": "v",
    "together/Qwen--Qwen2.5-72B-Instruct-Turbo": "2",
}


def augment_bdp_elicited_best(
    bdp: BacktestDataPoint[Frontier],
):
    # dummy implementation for now
    bdp.split_train = bdp.split_train.assign(
        elicited_best=0.0, elicited_all_frontier=0.0, elicited_distinct=0.0
    )
    bdp.split_test = bdp.split_test.assign(
        elicited_best=0.0, elicited_all_frontier=0.0, elicited_distinct=0.0
    )


def plot_figure_1(
    backtest: BacktestFrontierData,
    benchmarks: list[Figure1Benchmark],
    min_date: float = 2022,
    max_date: float = 2027,
    ci=0.95,
):
    assert (
        backtest.splitter.key == "release_date"
    ), "Only release_date supported for now"

    _, n_bench = backtest.results.shape
    for benchmark in benchmarks:
        assert benchmark.id < n_bench, f"Invalid benchmark_id {id}"

    fig, ax = plt.subplots(
        1, len(benchmarks), figsize=(3.5 * len(benchmarks), 3.5), squeeze=False
    )

    for j, benchmark in enumerate(benchmarks):
        benchmark_id = benchmark.id
        curr_ax = ax[0, j]
        curr_ax.grid()
        curr_ax.set_axisbelow(True)

        # first, plot the train points.
        # We need to use the final dataframe to get the ground truth data, since it will have the best-trained PC-1 score (and doesn't matter for the other models)
        bdp_g: BacktestDataPoint[Frontier] = backtest.global_split_results[benchmark_id]
        bdp_g_copy = bdp_g.copy()

        do_elicit = (
            isinstance(bdp_g.model, FrontierDateAllElicitedToEloPredictor)
            or isinstance(bdp_g.model, FrontierDateBestElicitedToEloPredictor)
            or isinstance(bdp_g.model, FrontierDateDistinctElicitedToEloPredictor)
        ) and benchmark.elicitations is not None

        augment_bdp_elicited_best(bdp_g_copy)

        augment_df_slaw(
            bdp_g_copy.model,
            bdp_g_copy.slaw,
            bdp_g_copy.split_train,
        )

        x_resolution = 100

        xs = np.linspace(min_date, max_date, x_resolution)

        n_samples = 10_000

        if benchmark.density:
            # first get a bunch of samples

            # samples = compute_density_samples_via_bootstrapping(
            #     bdp_g_copy.split_train,
            #     xs,
            #     bdp_g.slaw.benchmark,
            #     n_bootstrap=n_samples,
            #     fit_on_line=benchmark.fit_on_line,
            # )

            samples = compute_density_samples_via_bootstrapping(
                type(bdp_g.model),
                bdp_g_copy.split_train,
                xs,
                bdp_g.slaw.benchmark,
                canonical_slaw=bdp_g.slaw,
                extra_data=benchmark.elicitations,
                n_samples=n_samples,
                percentage_drawn=2 if do_elicit else 1,
            )

            # samples = compute_density_samples_linear(
            #     bdp_g_copy.split_train,
            #     xs,
            #     bdp_g.slaw.benchmark,
            #     bdp_g.slaw,
            #     n_samples=n_samples,
            # )

            # get the samples
            v1 = np.broadcast_to(xs, (samples.shape[0], len(xs))).flatten()
            v2 = samples.flatten()
            curr_ax.hist2d(
                v1,
                v2,
                bins=x_resolution,
                cmap="Blues",
                vmin=0,
                vmax=samples.shape[0] / 4,
            )
        else:
            # samples = compute_density_samples_via_bootstrapping(
            #     bdp_g_copy.split_train,
            #     xs,
            #     bdp_g.slaw.benchmark,
            #     n_bootstrap=n_samples,
            #     fit_on_line=benchmark.fit_on_line,
            # )

            samples = compute_density_samples_via_bootstrapping(
                type(bdp_g.model),
                bdp_g_copy.split_train,
                xs,
                bdp_g.slaw.benchmark,
                canonical_slaw=bdp_g.slaw,
                extra_data=benchmark.elicitations,
                n_samples=n_samples,
            )

            lower = np.quantile(samples, (1 - ci) / 2, axis=0)
            upper = np.quantile(samples, (1 + ci) / 2, axis=0)

            curr_ax.fill_between(
                xs,
                lower,
                upper,
                color="tab:blue",
                alpha=0.15,
            )

        frontier_df = get_running_top_n(
            bdp_g_copy.split_train,
            "release_date",
            bdp_g.slaw.benchmark,
            1,
            "model",
        )
        frontier_set = set(frontier_df["model"])
        non_frontier_df = bdp_g_copy.split_train[
            ~bdp_g_copy.split_train["model"].isin(frontier_set)
        ]

        df = bdp_g_copy.split_train

        # plot non-frontier points
        curr_ax.scatter(
            non_frontier_df["release_date"],
            non_frontier_df[bdp_g.slaw.benchmark],
            label="Low Capability Elicitation Models" if j == 0 else None,
            alpha=0.5,
            s=10,
            color="tab:blue",
        )

        # plot frontier points
        curr_ax.scatter(
            frontier_df["release_date"],
            frontier_df[bdp_g.slaw.benchmark],
            label="Low Capability Elicitation Models (Frontier)" if j == 0 else None,
            alpha=1,
            s=75,
            color="tab:blue",
            marker="*",
        )

        # # annotate frontier points
        # for i, row in frontier_df.iterrows():
        #     name = row["model"] if row["model"] not in friendly_names else friendly_names[row["model"]]
        #     curr_ax.annotate(
        #         name,
        #         (row["release_date"], row[bdp_g.slaw.benchmark]),
        #         textcoords="offset points",
        #         xytext=(-10, 5),
        #         ha="right",
        #         fontsize=7
        #     )

        xs_t = torch.tensor(xs, dtype=torch.float32).reshape(-1, 1)

        ys = (
            bdp_g_copy.slaw.forward(
                bdp_g_copy.model.predict_frontier_capability_scores(
                    torch.cat([xs_t, torch.zeros_like(xs_t)], dim=1)
                )
            )
            .detach()
            .numpy()
        )
        if do_elicit:
            ys_elicited = (
                bdp_g_copy.slaw.forward(
                    bdp_g_copy.model.predict_frontier_capability_scores(
                        torch.cat([xs_t, torch.ones_like(xs_t)], dim=1)
                    )
                )
                .detach()
                .numpy()
            )
        else:
            ys_elicited = None

        now = 2025.1
        now_idx = np.argmin(np.abs(xs - now))

        # plot conservative prediction
        curr_ax.plot(
            xs[:now_idx],
            ys[:now_idx],
            label="Low Capability Elicitation Forecast" if j == 0 else None,
            alpha=1,
            color="tab:blue",
        )
        curr_ax.plot(
            xs[now_idx:],
            ys[now_idx:],
            alpha=1,
            color="tab:blue",
            linestyle="--",
        )

        # plot elicited prediction
        if ys_elicited is not None and benchmark.elicitations is not None:
            curr_ax.plot(
                xs[:now_idx],
                ys_elicited[:now_idx],
                label="High Capability Elicitation Forecast" if j == 0 else None,
                alpha=1,
                color="tab:orange",
            )
            curr_ax.plot(
                xs[now_idx:],
                ys_elicited[now_idx:],
                alpha=1,
                color="tab:orange",
                linestyle="--",
            )

            if not benchmark.density:
                # plot the 95% confidence interval of the elicited forecast
                samples = compute_density_samples_via_bootstrapping(
                    type(bdp_g.model),
                    bdp_g_copy.split_train,
                    xs,
                    bdp_g.slaw.benchmark,
                    canonical_slaw=bdp_g.slaw,
                    extra_data=benchmark.elicitations,
                    n_samples=n_samples,
                    percentage_drawn=2 if do_elicit else 1,
                    elicit=True,
                )

                lower = np.quantile(samples, (1 - ci) / 2, axis=0)
                upper = np.quantile(samples, (1 + ci) / 2, axis=0)

                curr_ax.fill_between(
                    xs,
                    lower,
                    upper,
                    color="tab:orange",
                    alpha=0.15,
                )

            # plot all the points from the elicitation
            elicited_df = benchmark.elicitations

            # combine both unelicited and elicited data
            all_df = pd.concat([bdp_g_copy.split_train, elicited_df])
            all_df_frontier = get_running_top_n(
                all_df,
                "release_date",
                bdp_g.slaw.benchmark,
                1,
                "model",
            )

            frontier_set = set(all_df_frontier["model"])

            frontier_elicited_df = elicited_df[elicited_df["model"].isin(frontier_set)]
            non_frontier_elicited_df = elicited_df[
                ~elicited_df["model"].isin(frontier_set)
            ]

            curr_ax.scatter(
                non_frontier_elicited_df["release_date"],
                non_frontier_elicited_df[bdp_g.slaw.benchmark],
                label="Elicitations" if j == 0 else None,
                alpha=0.5,
                s=10,
                color="tab:orange",
            )
            curr_ax.scatter(
                frontier_elicited_df["release_date"],
                frontier_elicited_df[bdp_g.slaw.benchmark],
                label="Elicitations (Frontier)" if j == 0 else None,
                alpha=1,
                s=75,
                color="tab:orange",
                marker="*",
            )

            # plot arrows as well as o3
            for i, (model, score) in enumerate(
                bdp_g_copy.split_train.dropna(
                    subset=[f"{bdp_g_copy.slaw.benchmark} Elicited"]
                )[["model", f"{bdp_g_copy.slaw.benchmark} Elicited"]].values
            ):
                # get release date and unelicited score
                model_df = df[df["model"] == model]
                release_date = model_df["release_date"].values[0]
                unelicited_score = model_df[bdp_g.slaw.benchmark].values[0]

                # # draw vertical arrow
                # arrow_len = score - unelicited_score

                # # leave a little gap so we can see the arrow head
                # if arrow_len > 0.02:
                #     arrow_len -= 0.02
                # curr_ax.arrow(
                #     release_date,
                #     unelicited_score,
                #     0,
                #     arrow_len,
                #     head_width=0.1,
                #     head_length=0.02,
                #     length_includes_head=True,
                #     color="black",
                #     alpha=0.4,
                # )

                print(model)
                if model in ["o3", "claude-3.7-sonnet", "gpt-4.5-preview"]:
                    curr_ax.scatter(
                        [release_date],
                        [score],
                        label=(f"{model_friendly_names[model]}"),
                        color="tab:red",
                        s=40,
                        marker=model_markers[model],
                    )

        curr_ax.set_title(bdp_g.slaw.benchmark)
        curr_ax.set_xlabel("Release Date")
        curr_ax.set_ylabel(benchmark.yaxis)
        curr_ax.set_ylim(
            0.0,
            benchmark_ceiling_dict[bdp_g.slaw.benchmark],
        )
        # Shrink current axis's height by 10% on the bottom
        # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot
        box = curr_ax.get_position()
        curr_ax.set_position(
            [box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9]
        )

    fig.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=4)
    fig.tight_layout()
    return fig


# gets the split boundaries.
# if there are N splits, this returns N+2 values, where the first value is the minimum value of the key, and the last value is the maximum value of the key
def get_split_boundaries(
    b0: BacktestFrontierData, benchmark_id: int, padding: float = 0.0
):
    boundaries: list[float] = []
    for i, split_ in enumerate(b0.results[:, benchmark_id]):
        split: BacktestDataPoint[Frontier] = split_
        if i == 0:
            boundaries.append(split.split_train["release_date"].min())

        boundaries.append(split.split_train["release_date"].max())

        if i == len(b0.results[:, benchmark_id]) - 1:
            boundaries.append(split.split_test["release_date"].max())

    boundaries[0] -= padding
    boundaries[-1] += padding

    return boundaries


@dataclass
class CapabilityBacktest2:
    backtest: BacktestFrontierData


def plot_capability_backtesting_figure3(
    backtests: list[CapabilityBacktest2], benchmark_id: int
):

    benchmark = backtests[0].backtest.global_split_results[benchmark_id].slaw.benchmark

    prediction_title_dict = {
        "FrontierFlopToPC1Predictor": f"Flop $\\to$ PC-1 $\\to$ {benchmark}",
        "FrontierFlopToEloPredictor": f"Flop $\\to$ Elo $\\to$ {benchmark}",
        "FrontierDateToPC1Predictor": f"Date $\\to$ PC-1 $\\to$ {benchmark}",
        "FrontierDateToEloPredictor": f"Date $\\to$ Elo $\\to$ {benchmark}",
        "FrontierFlopPredictor": f"Flop $\\to$ {benchmark}",
        "FrontierDatePredictor": f"Date $\\to$ {benchmark}",
    }

    # Create figure with 2 rows and N columns
    n_rows = 2
    n_cols = len(backtests)
    fig, ax = plt.subplots(
        n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows), squeeze=False
    )

    # Common color map and global variables
    color_map = plt.get_cmap("coolwarm")
    max_rmse = 0  # For consistent ylim across histograms
    split_labels = []  # For storing split names once

    # First pass to calculate global maximum RMSE and split labels
    for capability_backtest in backtests:
        backtest = capability_backtest.backtest
        _, test_err = compute_test_train_error_frontier(backtest.results)
        current_rmse = np.sqrt(test_err[:, benchmark_id])
        max_rmse = max(max_rmse, current_rmse.max())

        # Generate split labels once using first backtest (assuming same splits)
        if not split_labels:
            n_splits = backtest.results.shape[0]
            split_labels = [f"Split {i+1}" for i in range(n_splits)]

    # Add 10% padding to max RMSE
    max_rmse *= 1.1

    # Get common color norm from first backtest (assuming same splits)
    base_backtest = backtests[0].backtest
    boundaries = np.array(
        get_split_boundaries(base_backtest, benchmark_id, padding=0.0)
    )
    line_color_norm = matplotlib.colors.BoundaryNorm(boundaries, color_map.N)
    scatter_color_norm = matplotlib.colors.BoundaryNorm(boundaries + 0.01, color_map.N)

    # Main plotting loop
    for j, capability_backtest in enumerate(backtests):
        backtest = capability_backtest.backtest

        # --- Calibration Plot (Top Row) ---
        curr_ax_calib = ax[0, j]
        _, test_err = compute_test_train_error_frontier(backtest.results)

        # Prepare data for the current benchmark
        bdp_g: BacktestDataPoint[Frontier] = backtest.global_split_results[benchmark_id]
        bdp_g_copy = bdp_g.copy()
        augment_train_test_slaw(
            bdp_g_copy.model,
            bdp_g_copy.slaw,
            bdp_g_copy.split_train,
            bdp_g_copy.split_test,
        )

        frontier_df = get_running_top_n(
            bdp_g_copy.split_train,
            "release_date",
            bdp_g.slaw.benchmark,
            1,
            "model",
        )

        frontier_set = set(frontier_df["model"])

        bdp_g_splits = list(backtest.splitter.split(bdp_g_copy.split_train))

        plotted_points = set()
        last_max_v = bdp_g_copy.split_train[backtest.splitter.key].min()
        for split_idx, (train, test) in enumerate(bdp_g_splits):
            max_v = train[backtest.splitter.key].max()
            min_v = last_max_v
            last_max_v = max_v

            df = train[~train[backtest.splitter.key].isin(plotted_points)]

            split_nonfrontier_df = df[~df["model"].isin(frontier_set)]
            split_frontier_df = df[df["model"].isin(frontier_set)]

            y_key = f"{bdp_g.slaw.benchmark}"

            # Plot current split's training points (non-frontier)
            curr_ax_calib.scatter(
                split_nonfrontier_df[y_key],
                split_nonfrontier_df[f"{y_key} pred"],
                c=split_nonfrontier_df["release_date"],
                cmap=color_map,
                norm=scatter_color_norm,
                alpha=0.5,
                s=15,
            )
            # plot frontier points now
            curr_ax_calib.scatter(
                split_frontier_df[y_key],
                split_frontier_df[f"{y_key} pred"],
                label=(
                    f"{year_formatter(min_v, None)} to {year_formatter(max_v, None)} (Split {split_idx})"
                    if j == 0
                    else None
                ),
                c=split_frontier_df["release_date"],
                cmap=color_map,
                norm=scatter_color_norm,
                marker="*",
                s=75,
            )
            plotted_points.update(train[backtest.splitter.key])

            # Handle last split's remaining points
            if split_idx == len(bdp_g_splits) - 1:
                df = bdp_g_copy.split_train[
                    ~bdp_g_copy.split_train[backtest.splitter.key].isin(plotted_points)
                ]

                split_nonfrontier_df = df[~df["model"].isin(frontier_set)]
                split_frontier_df = df[df["model"].isin(frontier_set)]

                y_key = f"{bdp_g.slaw.benchmark}"

                curr_ax_calib.scatter(
                    split_nonfrontier_df[y_key],
                    split_nonfrontier_df[f"{y_key} pred"],
                    c=split_nonfrontier_df["release_date"],
                    cmap=color_map,
                    norm=scatter_color_norm,
                    s=15,
                    alpha=0.5,
                )

                # plot frontier points now
                curr_ax_calib.scatter(
                    split_frontier_df[y_key],
                    split_frontier_df[f"{y_key} pred"],
                    label=(
                        f"{year_formatter(max_v, None)}+ (Split {split_idx+1})"
                        if j == 0
                        else None
                    ),
                    c=split_frontier_df["release_date"],
                    cmap=color_map,
                    norm=scatter_color_norm,
                    marker="*",
                    s=75,
                )

                # Plot diagonal line
                curr_ax_calib.plot(
                    [0.1, 0.7],
                    [0.1, 0.7],
                    color="black",
                    linestyle="--",
                )

        # Set titles and labels for calibration plot
        curr_ax_calib.set_title(
            prediction_title_dict.get(
                type(bdp_g.model).__name__, type(bdp_g.model).__name__
            ),
            usetex=True,
        )
        if j == 0:
            curr_ax_calib.set_ylabel(f"{bdp_g.slaw.benchmark} Predicted")
        curr_ax_calib.set_xlabel(f"{bdp_g.slaw.benchmark} Actual")

        # --- Histogram Plot (Bottom Row) ---
        curr_ax_hist = ax[1, j]
        _, test_err = compute_test_train_error_frontier(backtest.results)
        rmse_per_split = np.sqrt(test_err[:, benchmark_id])

        # Get max values for each split to determine colors
        max_values = [
            result.split_train[backtest.splitter.key].max()
            for result in backtest.results[:, benchmark_id]
        ]
        colors = color_map(line_color_norm(max_values))

        # Plot bars with consistent ylim
        x = np.arange(len(rmse_per_split))
        bars = curr_ax_hist.bar(x, rmse_per_split, color=colors)
        curr_ax_hist.set_ylim(0, max_rmse)

        # Configure histogram plot
        curr_ax_hist.set_xticks(x)
        curr_ax_hist.set_xticklabels(split_labels, rotation=0)
        if j == 0:
            curr_ax_hist.set_ylabel("Frontier RMSE")
        curr_ax_hist.grid(True, axis="y", alpha=0.3)

        # Add bar value labels
        for bar in bars:
            height = bar.get_height()
            curr_ax_hist.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    fig.subplots_adjust(bottom=0.15, hspace=0.3)

    # Add single shared colorbar below histograms
    cbar_ax = fig.add_axes((0.25, 0.06, 0.5, 0.015))  # (x, y, width, height) as tuple

    cbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=line_color_norm, cmap=color_map),
        cax=cbar_ax,
        orientation="horizontal",
    )
    cbar.ax.xaxis.set_major_formatter(year_formatter)
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.xaxis.set_label_position("top")

    # Final figure adjustments
    benchmark_name = (
        backtests[0]
        .backtest.results[0, benchmark_id]
        .slaw.benchmark.replace(" Raw", "")
    )
    fig.suptitle(f"Calibration and Test RMSE for {benchmark_name}", y=0.97)

    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.05), ncol=4)

    plt.show()

def plot_capability_backtesting_figure3_corrected(backtests: list[CapabilityBacktest2], benchmark_id: int):
    benchmark = backtests[0].backtest.global_split_results[benchmark_id].slaw.benchmark
    
    prediction_title_dict = {
        "FrontierFlopToPC1Predictor": f"Flop $\\to$ PC-1 $\\to$ {benchmark}",
        "FrontierFlopToEloPredictor": f"Flop $\\to$ Elo $\\to$ {benchmark}",
        "FrontierDateToPC1Predictor": f"Date $\\to$ PC-1 $\\to$ {benchmark}",
        "FrontierDateToEloPredictor": f"Date $\\to$ Elo $\\to$ {benchmark}",
        "FrontierFlopPredictor": f"Flop $\\to$ {benchmark}",
        "FrontierDatePredictor": f"Date $\\to$ {benchmark}",
    }


    markerlist = ["o", "x", "s", "D"]

    # Create figure with 2 rows and N columns
    n_rows = 2
    n_cols = len(backtests)
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows), squeeze=False)

    # Common color map and global variables
    color_map = plt.get_cmap("coolwarm")
    max_rmse = 0  # For consistent ylim across histograms
    split_labels = []  # For storing split names once

    # First pass to calculate global maximum RMSE and split labels
    for capability_backtest in backtests:
        backtest = capability_backtest.backtest
        _, test_err = compute_test_train_error_frontier(backtest.results)
        current_rmse = np.sqrt(test_err[:, benchmark_id])
        max_rmse = max(max_rmse, current_rmse.max())
        
        # Generate split labels once using first backtest (assuming same splits)
        if not split_labels:
            n_splits = backtest.results.shape[0]
            split_labels = [f"Split {i+1}" for i in range(n_splits)]

    # Add 10% padding to max RMSE
    max_rmse *= 1.1

    # Get common color norm from first backtest (assuming same splits)
    base_backtest = backtests[0].backtest
    boundaries = np.array(get_split_boundaries(base_backtest, benchmark_id, padding=0.0))
    line_color_norm = matplotlib.colors.BoundaryNorm(boundaries, color_map.N)
    scatter_color_norm = matplotlib.colors.BoundaryNorm(boundaries + 0.01, color_map.N)

    # Main plotting loop
    for j, capability_backtest in enumerate(backtests):
        backtest = capability_backtest.backtest


        # --- Calibration Plot (Top Row) ---
        curr_ax_calib = ax[0, j]
        _, test_err = compute_test_train_error_frontier(backtest.results)

        # Prepare data for the current benchmark
        bdp_g: BacktestDataPoint[Frontier] = backtest.global_split_results[benchmark_id]
        bdp_g_copy = bdp_g.copy()
        augment_train_test_slaw(
            bdp_g_copy.model,
            bdp_g_copy.slaw,
            bdp_g_copy.split_train,
            bdp_g_copy.split_test,
        )

        bdp_g_splits = list(backtest.splitter.split(bdp_g_copy.split_train))

        plotted_points = set()
        last_max_v = bdp_g_copy.split_train[backtest.splitter.key].min()
        for split_idx, (train, test) in enumerate(bdp_g_splits):
            max_v = train[backtest.splitter.key].max()
            min_v = last_max_v
            last_max_v = max_v

            df = train[~train[backtest.splitter.key].isin(plotted_points)]

            y_key = f"{bdp_g.slaw.benchmark}"
            true_y = df[y_key]
            pred_y = df[f"{y_key} pred"]

            # Plot current split's training points
            scatter = curr_ax_calib.scatter(
                true_y,
                pred_y,
                label=f"{year_formatter(min_v, None)} to {year_formatter(max_v, None)} (Split {split_idx})" if j == 0 else None,
                c=df["release_date"],
                cmap=color_map,
                norm=scatter_color_norm,
                marker=markerlist[split_idx % len(markerlist)],
                s=20,
            )
            plotted_points.update(train[backtest.splitter.key])

            # Handle last split's remaining points
            if split_idx == len(bdp_g_splits) - 1:
                df = bdp_g_copy.split_train[
                    ~bdp_g_copy.split_train[backtest.splitter.key].isin(plotted_points)
                ]
                true_y = df[y_key]
                pred_y = df[f"{y_key} pred"]
                curr_ax_calib.scatter(
                    true_y,
                    pred_y,
                    label=f"{year_formatter(max_v, None)}+ (Split {split_idx + 1})" if j == 0 else None,
                    c=df["release_date"],
                    cmap=color_map,
                    norm=scatter_color_norm,
                    marker=markerlist[(split_idx + 1) % len(markerlist)],
                    s=20,
                )

                # Plot diagonal line
                curr_ax_calib.plot(
                    [0.1, 0.7],
                    [0.1, 0.7],
                    color="black",
                    linestyle="--",
                )

        # Set titles and labels for calibration plot
        curr_ax_calib.set_title(
            prediction_title_dict.get(type(bdp_g.model).__name__, type(bdp_g.model).__name__)
        )
        if j == 0:
            curr_ax_calib.set_ylabel(bdp_g.slaw.benchmark.replace("Raw", "Predicted"))
        # curr_ax_calib.set_xlabel(bdp_g.slaw.benchmark.replace(" Raw", ""))


        # --- Histogram Plot (Bottom Row) ---
        curr_ax_hist = ax[1, j]
        _, test_err = compute_test_train_error_frontier(backtest.results)
        rmse_per_split = np.sqrt(test_err[:, benchmark_id])

        # Get max values for each split to determine colors
        max_values = [
            result.split_train[backtest.splitter.key].max()
            for result in backtest.results[:, benchmark_id]
        ]
        colors = color_map(line_color_norm(max_values))

        # Plot bars with consistent ylim
        x = np.arange(len(rmse_per_split))
        bars = curr_ax_hist.bar(x, rmse_per_split, color=colors)
        curr_ax_hist.set_ylim(0, max_rmse)

        # Configure histogram plot
        curr_ax_hist.set_xticks(x)
        curr_ax_hist.set_xticklabels(split_labels, rotation=0)
        if j == 0:
            curr_ax_hist.set_ylabel('RMSE')
        curr_ax_hist.grid(True, axis='y', alpha=0.3)

        # Add bar value labels
        for bar in bars:
            height = bar.get_height()
            curr_ax_hist.annotate(f'{height:.3f}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom')

    fig.subplots_adjust(bottom=0.15)

    # Add single shared colorbar below histograms
    cbar_ax = fig.add_axes(
        (0.25, 0.06, 0.5, 0.015)  # (x, y, width, height) as tuple
    )
    
    cbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=line_color_norm, cmap=color_map),
        cax=cbar_ax,
        orientation='horizontal',
    )
    cbar.ax.xaxis.set_major_formatter(year_formatter)
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')

    # Final figure adjustments
    benchmark_name = backtests[0].backtest.results[0, benchmark_id].slaw.benchmark.replace(" Raw", "")
    fig.suptitle(f"Calibration and Test RMSE for {benchmark_name}", y=0.97)
    
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.05), ncol=4)

    plt.show()
    

def plot_figure_3(
    train_df: pd.DataFrame,
    dataframe_benchmarks: list[str],
    predicted_benchmark: str,
):
    """
    (1) (2)
    (3) (4)

    1 = Release Date -> PC-1
    2 = log10 FLOP_opt -> PC-1
    3 = Release Date -> Elo
    4 = log10 FLOP_opt -> Elo
    """

    nxplots = 2
    nyplots = 2

    fig, ax = plt.subplots(
        nyplots, nxplots, figsize=(nxplots * 5.25, nyplots * 1.8), squeeze=False
    )

    _, train_df, _ = augment_train_test_pc1(
        dataframe_benchmarks,
        predicted_benchmark,
        train_df,
        None,
    )

    # these four are relatively simple to plot
    methods_to_plot = [
        (
            0,
            0,
            FrontierFlopToPC1Predictor,
        ),
        (
            0,
            1,
            FrontierDateToPC1Predictor,
        ),
        (
            1,
            0,
            FrontierFlopToEloPredictor,
        ),
        (
            1,
            1,
            FrontierDateToEloPredictor,
        ),
    ]

    for i, j, FrontierCls in methods_to_plot:
        x_key = FrontierCls.necessary_benchmarks()[0]

        # setup axis to not overlap
        if x_key == "release_date":
            loc = ticker.MultipleLocator(
                base=0.5
            )  # this locator puts ticks at regular intervals
            ax[i, j].xaxis.set_major_locator(loc)

        xs = np.array(train_df[x_key])
        frontier_df = get_running_top_n(
            train_df,
            x_key,
            FrontierCls.capability(),
            1,
            "model",
        )

        method = FrontierCls(
            torch.tensor(frontier_df[x_key].values, dtype=torch.float32).reshape(-1, 1),
            torch.tensor(
                frontier_df[FrontierCls.capability()].values, dtype=torch.float32
            ),
        )

        frontier_set = set(frontier_df["model"])
        non_frontier_df = train_df[~train_df["model"].isin(frontier_set)]

        # plot linear fit
        xlinspace = np.linspace(xs.min(), xs.max(), 100)
        ylinspace = (
            method.predict_frontier_capability_scores(
                torch.tensor(xlinspace, dtype=torch.float32).reshape(-1, 1)
            )
            .detach()
            .numpy()
        )

        # compute pearson correlation
        r2 = r2_score(
            frontier_df[FrontierCls.capability()],
            method.predict_frontier_capability_scores(
                torch.tensor(frontier_df[x_key].values, dtype=torch.float32).reshape(
                    -1, 1
                )
            )
            .detach()
            .numpy(),
        )

        # add text of the equation to the upper left corner
        custom_line = [
            Line2D(
                [0],
                [0],
                color="tab:red",
                label=f"y = {method.beta[0].item():.2f}x + {method.beta[1].item():.2f}\n$R^2$ = {r2:.2f}",
            )
        ]
        ax[i, j].legend(handles=custom_line, prop={"size": 8}, loc="lower right")

        ax[i, j].plot(
            xlinspace,
            ylinspace,
            label=f"Best Fit on only Frontier Models" if i == 0 and j == 0 else None,
            color="tab:red",
        )

        # plot non-frontier models
        ax[i, j].scatter(
            non_frontier_df[x_key],
            non_frontier_df[FrontierCls.capability()],
            label="Open Source Models" if i == 0 and j == 0 else None,
            color="tab:green",
            alpha=0.5,
            s=10,
        )

        # plot frontier models
        ax[i, j].scatter(
            frontier_df[x_key],
            frontier_df[FrontierCls.capability()],
            label="Open Source Models (Frontier)" if i == 0 and j == 0 else None,
            color="tab:green",
            marker="*",
            s=75,
        )

        # change size of xticks and yticks to be small
        ax[i, j].tick_params(axis="both", which="major", labelsize=7)

    cols = ["log-FLOP (1E21)", "Release Date"]
    rows = ["PC-1", "Elo"]

    # set column names
    for a, col in zip(ax[1], cols):
        a.set_xlabel(col, size="medium")

    # remove xticks for the top row
    for a, col in zip(ax[0], cols):
        a.set_xticks([])

    # set row names
    for a, row in zip(ax[:, 0], rows):
        a.set_ylabel(row, rotation=0, size="medium")
        a.yaxis.set_label_coords(-0.175, 0.4)

    # set formatter for release date
    for a in ax[:, 1]:
        a.xaxis.set_major_formatter(ticker.FuncFormatter(year_formatter))

    fig.suptitle("Input Variables vs PC-1 and Elo", size="large")
    fig.subplots_adjust(bottom=0.2)
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.04), ncol=4)


def compare(
    data1: BacktestFrontierData,
    data2: BacktestFrontierData,
):
    plot_errmatrix_comparison([data1, data2])

    print(f"{data1.model_class.__name__} Train Error: {data1.results.mean()}")
    print(f"{data2.model_class.__name__} Test Error: {data2.results.mean()}")

    print(
        f"Train Percentage Improvement: {(data1.results.mean() - data2.results.mean()) / data1.results.mean() * 100:.2f}%"
    )
    print(
        f"Test Percentage Improvement: {(data1.results.mean() - data2.results.mean()) / data1.results.mean() * 100:.2f}%"
    )


# %%

#####################################
# Train and fit global linear and logit models of PC-1
# Expanding Window
#####################################

ewbs = ExpandingWindowBacktestSplitter(
    min_train_size=9,
    test_size=9,
    increment=9,
    key="release_date",
)


# %%
ewbs_frontier_flop_to_elo_data = backtest_models_frontier(
    ewbs, FrontierFlopToEloPredictor, openllm_elo_merged, openllm_elo_benchmarks
)
ewbs_frontier_flop_to_elo_train_err, ewbs_frontier_flop_to_elo_test_err = (
    compute_test_train_error_frontier(ewbs_frontier_flop_to_elo_data.results)
)

# %%
ewbs_frontier_date_to_elo_data = backtest_models_frontier(
    ewbs, FrontierDateToEloPredictor, openllm_elo_merged, openllm_elo_benchmarks
)
ewbs_frontier_date_to_elo_train_err, ewbs_frontier_date_to_elo_test_err = (
    compute_test_train_error_frontier(ewbs_frontier_date_to_elo_data.results)
)

# %%
ewbs_frontier_flop_to_pc1_data = backtest_models_frontier(
    ewbs, FrontierFlopToPC1Predictor, openllm_elo_merged, openllm_elo_benchmarks
)
ewbs_frontier_flop_to_pc1_train_err, ewbs_frontier_flop_to_pc1_test_err = (
    compute_test_train_error_frontier(ewbs_frontier_flop_to_pc1_data.results)
)

# %%
ewbs_frontier_date_to_pc1_data = backtest_models_frontier(
    ewbs, FrontierDateToPC1Predictor, openllm_elo_merged, openllm_elo_benchmarks
)
ewbs_frontier_date_to_pc1_train_err, ewbs_frontier_date_to_pc1_test_err = (
    compute_test_train_error_frontier(ewbs_frontier_date_to_pc1_data.results)
)


# %%
ewbs_frontier_flop_data = backtest_models_frontier(
    ewbs, FrontierFlopPredictor, openllm_elo_merged, openllm_elo_benchmarks
)
ewbs_frontier_flop_train_err, ewbs_frontier_flop_test_err = (
    compute_test_train_error_frontier(ewbs_frontier_flop_data.results)
)

# %%
ewbs_frontier_date_data = backtest_models_frontier(
    ewbs, FrontierDatePredictor, openllm_elo_merged, openllm_elo_benchmarks
)
ewbs_frontier_date_train_err, ewbs_frontier_date_test_err = (
    compute_test_train_error_frontier(ewbs_frontier_date_data.results)
)


# %%

# print ALL of the average errors:
#  FlopToElo, DateToElo, FlopToPC1, DateToPC1, FlopDateToElo, FlopDateToPC1

print(f"Flop -> Downstream Train MSE: {ewbs_frontier_flop_train_err.mean():.3f}")
print(f"Date -> Downstream Train MSE: {ewbs_frontier_date_train_err.mean():.3f}")
print(
    f"FlopToElo -> Downstream Train MSE: {ewbs_frontier_flop_to_elo_train_err.mean():.3f}"
)
print(
    f"DateToElo -> Downstream Train MSE: {ewbs_frontier_date_to_elo_train_err.mean():.3f}"
)

print(
    f"FlopToPC1 -> Downstream Train MSE: {ewbs_frontier_flop_to_pc1_train_err.mean():.3f}"
)
print(
    f"DateToPC1 -> Downstream Train MSE: {ewbs_frontier_date_to_pc1_train_err.mean():.3f}"
)


print()

print(f"Flop -> Downstream Test MSE: {ewbs_frontier_flop_test_err.mean():.3f}")
print(f"Date -> Downstream Test MSE: {ewbs_frontier_date_test_err.mean():.3f}")
print(
    f"FlopToElo -> Downstream Test MSE: {ewbs_frontier_flop_to_elo_test_err.mean():.3f}"
)
print(
    f"DateToElo -> Downstream Test MSE: {ewbs_frontier_date_to_elo_test_err.mean():.3f}"
)
print(
    f"FlopToPC1 -> Downstream Test MSE: {ewbs_frontier_flop_to_pc1_test_err.mean():.3f}"
)
print(
    f"DateToPC1 -> Downstream Test MSE: {ewbs_frontier_date_to_pc1_test_err.mean():.3f}"
)

# print RMSE
print()
print()

print(f"Flop -> Downstream Train RMSE: {ewbs_frontier_flop_train_err.mean()**0.5:.3f}")
print(f"Date -> Downstream Train RMSE: {ewbs_frontier_date_train_err.mean()**0.5:.3f}")
print(
    f"FlopToElo -> Downstream Train RMSE: {ewbs_frontier_flop_to_elo_train_err.mean()**0.5:.3f}"
)
print(
    f"DateToElo -> Downstream Train RMSE: {ewbs_frontier_date_to_elo_train_err.mean()**0.5:.3f}"
)
print(
    f"FlopToPC1 -> Downstream Train RMSE: {ewbs_frontier_flop_to_pc1_train_err.mean()**0.5:.3f}"
)
print(
    f"DateToPC1 -> Downstream Train RMSE: {ewbs_frontier_date_to_pc1_train_err.mean()**0.5:.3f}"
)

print()

print(f"Flop -> Downstream Test RMSE: {ewbs_frontier_flop_test_err.mean()**0.5:.3f}")
print(f"Date -> Downstream Test RMSE: {ewbs_frontier_date_test_err.mean()**0.5:.3f}")
print(
    f"FlopToElo -> Downstream Test RMSE: {ewbs_frontier_flop_to_elo_test_err.mean()**0.5:.3f}"
)
print(
    f"DateToElo -> Downstream Test RMSE: {ewbs_frontier_date_to_elo_test_err.mean()**0.5:.3f}"
)
print(
    f"FlopToPC1 -> Downstream Test RMSE: {ewbs_frontier_flop_to_pc1_test_err.mean()**0.5:.3f}"
)
print(
    f"DateToPC1 -> Downstream Test RMSE: {ewbs_frontier_date_to_pc1_test_err.mean()**0.5:.3f}"
)

# # test averaged results

# ewbs_avgd_train_err, ewbs_avgd_test_err = (
#     compute_test_train_error_frontier_avg(
#         [ewbs_frontier_date_to_pc1_data.results,
#          ewbs_frontier_date_to_elo_data.results,
#          ])
# )

# print(f"Average Train RMSE: {ewbs_avgd_train_err.mean()**0.5:.3f}")
# print(f"Average Test RMSE: {ewbs_avgd_test_err.mean()**0.5:.3f}")



# %%
plot_comparison(
    [
        ewbs_frontier_date_to_elo_data,
    ]
)

# %%

# compare the models

plot_comparison(
    [
        ewbs_frontier_flop_to_elo_data,
        ewbs_frontier_date_to_elo_data,
        ewbs_frontier_flop_to_pc1_data,
        ewbs_frontier_date_to_pc1_data,
    ],
    expand=True,
)


# %%

plot_errmatrix_comparison(
    [
        ewbs_frontier_flop_data,
        ewbs_frontier_date_data,
        ewbs_frontier_flop_to_elo_data,
        ewbs_frontier_date_to_elo_data,
        ewbs_frontier_flop_to_pc1_data,
        ewbs_frontier_date_to_pc1_data,
    ]
)


# %%

plot_split(
    ewbs_frontier_date_to_elo_data,
    2,
    "release_date",
    expand=False,
    line=True,
    capability=False,
)


# %%

plot_split(
    ewbs_frontier_date_data,
    5,
    "release_date",
    expand=False,
    line=True,
    capability=False,
)


# %%
plot_capability_backtesting_figure3(
    [
        CapabilityBacktest2(ewbs_frontier_flop_data),
        CapabilityBacktest2(ewbs_frontier_date_data),
        CapabilityBacktest2(ewbs_frontier_flop_to_pc1_data),
        CapabilityBacktest2(ewbs_frontier_flop_to_elo_data),
        CapabilityBacktest2(ewbs_frontier_date_to_pc1_data),
        CapabilityBacktest2(ewbs_frontier_date_to_elo_data),
    ],
    5,
)

# %%
plot_capability_backtesting_figure3_corrected(
    [
        CapabilityBacktest2(ewbs_frontier_flop_data),
        CapabilityBacktest2(ewbs_frontier_date_data),
        CapabilityBacktest2(ewbs_frontier_flop_to_pc1_data),
        CapabilityBacktest2(ewbs_frontier_flop_to_elo_data),
        CapabilityBacktest2(ewbs_frontier_date_to_pc1_data),
        CapabilityBacktest2(ewbs_frontier_date_to_elo_data),
    ],
    5,
)


# %%

plot_all_loss_curves(ewbs_frontier_date_to_elo_data)


# %%

############################################
# 2 x 2 Figure for Methods Section
###########################################

plot_figure_3(
    openllm_elo_merged,
    openllm_elo_benchmarks,
    "MMLU-PRO",
)

# %%

############################################
# Agentic Models
############################################

agentic_ewbs = ExpandingWindowBacktestSplitter(
    min_train_size=7,
    test_size=4,
    increment=10,
    key="release_date",
)


# %%
agentic_ewbs_frontier_date_elicited_to_elo_data = backtest_models_frontier(
    agentic_ewbs,
    FrontierDateDistinctElicitedToEloPredictor,
    agentic_benchmark,
    agentic_benchmark_benchmarks,
    dataframe_benchmark_elicitations=[swebench_scaffolds, cybench_scaffolds, None],
    fit_on_line=False,
)

# %%

fig1 = plot_figure_1(
    agentic_ewbs_frontier_date_elicited_to_elo_data,
    [
        Figure1Benchmark(
            0,
            "Success Rate",
            elicitations=swebench_scaffolds,
            density=False,
            fit_on_line=False,
        ),
        Figure1Benchmark(
            1, "Success Rate", elicitations=cybench_scaffolds, density=False
        ),
        Figure1Benchmark(2, "Average Normalized Score", density=False),
    ],
)


# %%
hist_fig = plot_histogram_for_all(
    agentic_ewbs_frontier_date_elicited_to_elo_data,
    [
        HistogramFigure(0, 0.9, elicitations=swebench_scaffolds),
        HistogramFigure(1, 0.9, elicitations=cybench_scaffolds),
        HistogramFigure(2, 1),
    ],
)


# %%

# Print the value of all 3 benchmarks at 2026
for elicitation in [0, 1]:
    for bdp in agentic_ewbs_frontier_date_elicited_to_elo_data.global_split_results:
        cap_score = bdp.model.predict_frontier_capability_scores(
            torch.tensor([2026, elicitation], dtype=torch.float32)
        )

        benchmark_val = bdp.slaw.forward(cap_score).item()

        print(
            f"{bdp.slaw.benchmark} at 2026: {benchmark_val:.3f} (Elicitation {elicitation})"
        )
    print()


# %%

# dump some json files
agentic_benchmark.to_json(
    "../website/src/assets/data/agentic_benchmark.json", orient="records", index=False
)
cybench_scaffolds.to_json(
    "../website/src/assets/data/cybench_scaffolds.json", orient="records", index=False
)
swebench_scaffolds.to_json(
    "../website/src/assets/data/swebench_scaffolds.json", orient="records", index=False
)

# %%
# dump density predictions of each benchmark as a json array of arrays

# we want 20 ticks per year

x_linspace = np.linspace(2022, 2035, (2035 - 2022) * 20)
y_linspace1 = np.linspace(0, 1, 100)
y_linspace16 = np.linspace(0, 1.66, 100)

swebench_forecast, swebench_forecast_elicited = compute_forecast_and_density(
    agentic_ewbs_frontier_date_elicited_to_elo_data,
    0,
    x_linspace,
    y_linspace1,
    elicitations=swebench_scaffolds,
)
assert swebench_forecast_elicited is not None
cybench_forecast, cybench_forecast_elicited = compute_forecast_and_density(
    agentic_ewbs_frontier_date_elicited_to_elo_data,
    1,
    x_linspace,
    y_linspace1,
    elicitations=cybench_scaffolds,
)
assert cybench_forecast_elicited is not None
rebench_forecast, _ = compute_forecast_and_density(
    agentic_ewbs_frontier_date_elicited_to_elo_data,
    2,
    x_linspace,
    y_linspace16,
)

Path("./jsons/swebench_forecast.json").write_text(
    swebench_forecast.model_dump_json()
)
Path("./jsons/swebench_forecast_elicited.json").write_text(
    swebench_forecast_elicited.model_dump_json()
)
Path("./jsons/cybench_forecast.json").write_text(
    cybench_forecast.model_dump_json()
)
Path("./jsons/cybench_forecast_elicited.json").write_text(
    cybench_forecast_elicited.model_dump_json()
)
Path("./jsons/rebench_forecast.json").write_text(
    rebench_forecast.model_dump_json()
)