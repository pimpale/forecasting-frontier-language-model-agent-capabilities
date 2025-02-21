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
            alpha=0.5,
            color=e.color,
        )
        ax.scatter(
            test_df[x_key],
            test_df[e.y_key],
            marker="o",
            label=e.y_label,
            alpha=0.5,
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


def augment_df_linear(linear_obs_model: LinearPC1Predictor, df_to_augment: pd.DataFrame):
    x = torch.tensor(df_to_augment[linear_obs_model.benchmarks].values, dtype=torch.float32)
    capability_score = linear_obs_model.predict_capability_scores_from_model_scores(x)
    x_hat = linear_obs_model.predict_benchmark_scores_from_capability_scores(capability_score)

    df_to_augment["PC-1 (linear)"] = capability_score.detach().numpy()

    for b_idx, benchmark in enumerate(linear_obs_model.benchmarks):
        df_to_augment[f"{benchmark} pred"] = x_hat.T[b_idx].detach().numpy()


def augment_train_test_linear(
    linear_obs_model: LinearPC1Predictor,
    train: pd.DataFrame,
    test: pd.DataFrame,
):
    augment_df_linear(linear_obs_model, train)
    augment_df_linear(linear_obs_model, test)


def augment_df_slaw(slaw: ScalingLaw, model: ObsScalingLawPredictor, df_to_augment: pd.DataFrame):
    model_scores = torch.tensor(df_to_augment[model.benchmarks].values, dtype=torch.float32)
    benchmark_scores = torch.tensor(df_to_augment[slaw.benchmark].values, dtype=torch.float32)
    capability_scores = model.predict_capability_scores_from_model_scores(model_scores).detach()

    df_to_augment[f"{model.intermediate} pred"] = capability_scores.numpy()

    df_to_augment[f"{slaw.benchmark} logit"] = (
        slaw.predict_logit_scores(benchmark_scores).detach().numpy()
    )
    df_to_augment[f"{slaw.benchmark} logit pred"] = (
        slaw.predict_benchmark_logit_scores(capability_scores).detach().numpy()
    )
    df_to_augment[f"{slaw.benchmark} pred"] = slaw.forward(capability_scores).detach().numpy()


def augment_train_test_slaw(
    slaw: ScalingLaw,
    model: ObsScalingLawPredictor,
    train: pd.DataFrame,
    test: pd.DataFrame,
):
    augment_df_slaw(slaw, model, train)
    augment_df_slaw(slaw, model, test)


@dataclass
class BacktestDataPoint[T: ObsScalingLawPredictor]:
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
class BacktestData:
    splitter: BacktestSplitter
    model_class: type[ObsScalingLawPredictor]
    benchmarks: list[str]
    splits: list[int]
    # 2D array of BacktestDataPoint on the splits x benchmarks
    results: npt.NDArray[np.object_]
    # 1D array of BacktestDataPoint on the benchmarks (using all points)
    global_split_results: npt.NDArray[np.object_]


def get_benchmark_list(
    ModelCls: type[ObsScalingLawPredictor],
    predicted_benchmark: str,
    dataframe_benchmarks: list[str],
) -> list[str]:
    maybe_fixed_benchmarks = ModelCls.fixed_benchmarks()
    if maybe_fixed_benchmarks is not None:
        benchmark_list = maybe_fixed_benchmarks
    else:
        benchmark_list = ModelCls.necessary_benchmarks() + [
            b for b in dataframe_benchmarks if b != predicted_benchmark
        ]

    return benchmark_list


def backtest_models_metric(
    splitter: BacktestSplitter,
    ModelCls: type[ObsScalingLawPredictor],
    dataframe: pd.DataFrame,
    dataframe_benchmarks: list[str],
) -> BacktestData:
    # create object ndarray

    train_test_splits = list(splitter.split(dataframe))

    data = BacktestData(
        splitter=splitter,
        model_class=ModelCls,
        benchmarks=dataframe_benchmarks,
        splits=[i for i in range(len(train_test_splits))],
        results=np.empty((len(train_test_splits), len(dataframe_benchmarks)), dtype=np.object_),
        global_split_results=np.empty(len(dataframe_benchmarks), dtype=np.object_),
    )

    n_trains = (len(train_test_splits) + 1) * len(dataframe_benchmarks)

    for split_idx, (train, test) in enumerate([(dataframe, dataframe.head(0))] + train_test_splits):
        for bench_idx, predicted_benchmark in enumerate(dataframe_benchmarks):
            i_train = split_idx * len(dataframe_benchmarks) + bench_idx
            print(f"Training {i_train}/{n_trains}")

            # construct the model inputs
            benchmark_list = get_benchmark_list(ModelCls, predicted_benchmark, dataframe_benchmarks)

            model_scores = torch.tensor(train[benchmark_list].values, dtype=torch.float32)

            # create model
            model = ModelCls(
                benchmark_list,
                benchmark_floors=[benchmark_floor_dict[b] for b in benchmark_list],
                train_model_scores=model_scores,
            )

            # train
            t0 = time.time()
            model.fit()
            model.eval()
            print(f"{ModelCls.__name__} Training Time: {time.time() - t0:.2f} seconds")

            # predict the excluded benchmark
            capability_scores = model.predict_capability_scores_from_model_scores(
                model_scores
            ).detach()
            benchmark_scores = torch.tensor(train[predicted_benchmark].values, dtype=torch.float32)
            slaw = ScalingLaw(
                benchmark=predicted_benchmark,
                floor=benchmark_floor_dict[predicted_benchmark],
                capability_scores=capability_scores,
                benchmark_scores=benchmark_scores,
            )
            t0 = time.time()
            slaw.fit()
            slaw.eval()
            print(f"Scaling Law Training Time: {time.time() - t0:.2f} seconds")

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


def year_formatter(x: float, _):
    year = int(x)
    month = int((x % 1) * 12 + 1)
    return f"{year}-{month:02d}"


def compute_test_train_error(
    arr: npt.NDArray[np.object_],
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
]:
    train_err = np.zeros_like(arr, dtype=np.float32)
    test_err = np.zeros_like(arr, dtype=np.float32)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            bdp: BacktestDataPoint[ObsScalingLawPredictor] = arr[i, j]
            train = bdp.split_train
            test = bdp.split_test
            slaw = bdp.slaw
            model = bdp.model

            for dataset, dataset_err_arr in ((train, train_err), (test, test_err)):
                x = torch.tensor(dataset[model.benchmarks].values, dtype=torch.float32)
                y = torch.tensor(dataset[slaw.benchmark].values, dtype=torch.float32)
                capability_scores = model.predict_capability_scores_from_model_scores(x)
                y_hat = slaw.forward(capability_scores)
                dataset_err_arr[i, j] = F.mse_loss(
                    y,
                    y_hat,
                ).item()

    return train_err, test_err


def plot_comparison(backtests: list[BacktestData], expand=False):
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
                    b0dp: BacktestDataPoint[ObsScalingLawPredictor] = b0.results[
                        split_idx, bench_idx
                    ]
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
                bdp: BacktestDataPoint[ObsScalingLawPredictor] = b.results[split_idx, bench_idx]
                bdp_copy = bdp.copy()
                augment_train_test_slaw(
                    bdp_copy.slaw,
                    bdp_copy.model,
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


def plot_split(backtest: BacktestData, benchmark_id: int, x_key: str, expand=False, line=False):
    color_list = [
        "tab:blue",
        "tab:cyan",
        "tab:green",
        "tab:orange",
    ]

    n_split, n_bench = backtest.results.shape
    assert benchmark_id < n_bench

    if expand:
        fig, ax = plt.subplots(1, n_split, figsize=(5 * n_split, 5), squeeze=False)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), squeeze=False)

    # first, plot the train points.
    # We need to use the final dataframe to get the ground truth data, since it will have the best-trained PC-1 score (and doesn't matter for the other models)
    bdp_g: BacktestDataPoint[ObsScalingLawPredictor] = backtest.global_split_results[benchmark_id]
    bdp_g_copy = bdp_g.copy()
    augment_train_test_slaw(
        bdp_g_copy.slaw,
        bdp_g_copy.model,
        bdp_g_copy.split_train,
        bdp_g_copy.split_test,
    )

    bdp_g_splits = list(backtest.splitter.split(bdp_g_copy.split_train))

    for j in range(len(bdp_g_splits) if expand else 1):
        curr_ax = ax[0, j]

        plotted_points = set()
        last_max_v = bdp_g_copy.split_train[backtest.splitter.key].min()
        for i, (train, test) in enumerate(bdp_g_splits):
            max_v = train[backtest.splitter.key].max()
            min_v = last_max_v
            last_max_v = max_v

            df = train[~train[backtest.splitter.key].isin(plotted_points)]

            y_key = f"{bdp_g.slaw.benchmark}"

            curr_ax.scatter(
                df[x_key],
                df[y_key],
                label=f"{min_v:.1f} - {max_v:.1f} {backtest.splitter.key}",
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
                    color=color_list[len(bdp_g_splits)],
                )
                curr_ax.set_title(f"{x_key} vs {y_key}")
                curr_ax.set_xlabel(x_key)
                curr_ax.set_ylabel(y_key)

    # now plot the predictions
    # to do this, we use the model to make predictions for the entire space and plot it

    for split_idx in range(n_split):
        color = color_list[split_idx]
        bdp: BacktestDataPoint[ObsScalingLawPredictor] = backtest.results[split_idx, benchmark_id]

        # augment the global split with the model's predictions
        bdp_g_copy2 = bdp_g.copy()
        augment_df_slaw(bdp.slaw, bdp.model, bdp_g_copy2.split_train)

        if expand:
            curr_ax = ax[0, split_idx]
        else:
            curr_ax = ax[0, 0]

        # plot the predictions
        if line:
            xs = np.array(bdp_g_copy.split_train[x_key])
            ys = np.array(bdp_g_copy2.split_train[f"{bdp.slaw.benchmark} pred"])

            # Sort both arrays based on x values
            sort_idx = np.argsort(xs)

            curr_ax.plot(
                xs[sort_idx],
                ys[sort_idx],
                label=f"{type(bdp.model).__name__} pred",
                alpha=1,
                color=color,
            )
        else:
            curr_ax.scatter(
                bdp_g_copy.split_train[x_key],
                bdp_g_copy2.split_train[f"{bdp.slaw.benchmark} pred"],
                label=f"{type(bdp.model).__name__} pred",
                alpha=1,
                marker="x",
                color=color,
            )
        curr_ax.legend()

    fig.tight_layout()
    plt.show()


# gets the split boundaries.
# if there are N splits, this returns N+2 values, where the first value is the minimum value of the key, and the last value is the maximum value of the key
def get_split_boundaries(b0: BacktestData, benchmark_id: int, padding: float = 0.0):
    boundaries: list[float] = []
    for i, split_ in enumerate(b0.results[:, benchmark_id]):
        split: BacktestDataPoint[ObsScalingLawPredictor] = split_
        if i == 0:
            boundaries.append(split.split_train["release_date"].min())

        boundaries.append(split.split_train["release_date"].max())

        if i == len(b0.results[:, benchmark_id]) - 1:
            boundaries.append(split.split_test["release_date"].max())

    boundaries[0] -= padding
    boundaries[-1] += padding

    return boundaries


@dataclass
class CapabilityBacktest:
    backtest: BacktestData
    extrapolatable: bool = True


def plot_capability_backtesting_figure(backtests: list[CapabilityBacktest], benchmark_id: int):
    prediction_title_dict = {
        "PC-1": "PC-1",
        "Elo": "Elo",
        "log10 FLOP_opt": "log-FLOP (1E21)",
        "release_date": "Release Date",
    }

    markerlist = ["o", "x", "s", "D"]

    b0 = backtests[0].backtest
    boundaries = np.array(get_split_boundaries(b0, benchmark_id, padding=0.0))

    color_map = plt.get_cmap("coolwarm")
    line_color_norm = matplotlib.colors.BoundaryNorm(boundaries, color_map.N)
    # shift the boundaries slightly forward so that items on the line are grouped in the previous split
    scatter_color_norm = matplotlib.colors.BoundaryNorm(boundaries + 0.01, color_map.N)

    n_split, n_bench = backtests[0].backtest.results.shape
    assert benchmark_id < n_bench

    fig, ax = plt.subplots(1, len(backtests), figsize=(3 * len(backtests), 3.5), squeeze=False)

    for j, capability_backtest in enumerate(backtests):
        backtest = capability_backtest.backtest
        extrapolatable = capability_backtest.extrapolatable
        _, test_err = compute_test_train_error(backtest.results)

        # first, plot the train points.
        # We need to use the final dataframe to get the ground truth data, since it will have the best-trained PC-1 score (and doesn't matter for the other models)
        bdp_g: BacktestDataPoint[ObsScalingLawPredictor] = backtest.global_split_results[
            benchmark_id
        ]
        bdp_g_copy = bdp_g.copy()
        augment_train_test_slaw(
            bdp_g_copy.slaw,
            bdp_g_copy.model,
            bdp_g_copy.split_train,
            bdp_g_copy.split_test,
        )

        bdp_g_splits = list(backtest.splitter.split(bdp_g_copy.split_train))

        curr_ax = ax[0, j]

        plotted_points = set()
        last_max_v = bdp_g_copy.split_train[backtest.splitter.key].min()
        for i, (train, test) in enumerate(bdp_g_splits):
            max_v = train[backtest.splitter.key].max()
            min_v = last_max_v
            last_max_v = max_v

            df = train[~train[backtest.splitter.key].isin(plotted_points)]

            y_key = f"{bdp_g.slaw.benchmark}"

            curr_ax.scatter(
                df[f"{bdp_g.model.intermediate} pred"],
                df[y_key],
                label=f"{min_v:.1f} - {max_v:.1f} (Split {i})" if j == 0 else None,
                c=df["release_date"],
                cmap=color_map,
                norm=scatter_color_norm,
                marker=markerlist[i],
                s=20,
            )

            plotted_points.update(train[backtest.splitter.key])

            if i == len(bdp_g_splits) - 1:
                df = bdp_g_copy.split_train[
                    ~bdp_g_copy.split_train[backtest.splitter.key].isin(plotted_points)
                ]
                curr_ax.scatter(
                    df[f"{bdp_g.model.intermediate} pred"],
                    df[y_key],
                    label=f"{max_v:.1f}+ (Split {i+1})" if j == 0 else None,
                    c=df["release_date"],
                    cmap=color_map,
                    norm=scatter_color_norm,
                    marker=markerlist[i + 1],
                    s=20,
                )

                curr_ax.set_xlabel(
                    (
                        prediction_title_dict[bdp_g.model.intermediate]
                        if bdp_g.model.intermediate in prediction_title_dict
                        else bdp_g.model.intermediate
                    ),
                    size="large",
                )
                if j == 0:
                    curr_ax.set_ylabel(bdp_g.slaw.benchmark)

        # now plot the predictions
        # to do this, we use the model to make predictions for the entire space and plot it

        lines = []
        for split_idx in range(n_split):
            bdp: BacktestDataPoint[ObsScalingLawPredictor] = backtest.results[
                split_idx, benchmark_id
            ]
            # sample color from the viridis colormap
            color = color_map(line_color_norm(bdp.split_train["release_date"].max()))

            # plot the predictions
            if extrapolatable:
                x_linspace = np.linspace(
                    bdp_g_copy.split_train[f"{bdp_g.model.intermediate} pred"].min(),
                    bdp_g_copy.split_train[f"{bdp_g.model.intermediate} pred"].max(),
                    100,
                )
                y = bdp.slaw(torch.tensor(x_linspace).reshape(-1, 1)).detach().numpy()

                lines += curr_ax.plot(
                    x_linspace,
                    y,
                    label=f"Forecast using split {split_idx}" if j == 0 else None,
                    alpha=1,
                    color=color,
                )
            else:
                # augment the global split with the model's predictions
                bdp_g_copy2 = bdp_g.copy()
                augment_df_slaw(bdp.slaw, bdp.model, bdp_g_copy2.split_train)

                xs = np.array(bdp_g_copy.split_train[f"{bdp_g.model.intermediate} pred"])
                ys = np.array(bdp_g_copy2.split_train[f"{bdp.slaw.benchmark} pred"])

                # Sort both arrays based on x values
                sort_idx = np.argsort(xs)

                lines += curr_ax.plot(
                    xs[sort_idx],
                    ys[sort_idx],
                    label=(f"Forecast predicting Split {split_idx+1}" if j == 0 else None),
                    alpha=1,
                    color=color,
                )

            curr_ax.legend(
                lines,
                [
                    f"Split {i+1} RMSE: {e:.3f}"
                    for i, e in enumerate(np.sqrt(test_err[:, benchmark_id]))
                ],
                prop={"size": "small"},
            )

        # set xticks if the splitter key is a date
        if bdp_g.model.intermediate == "release_date":
            loc = plticker.MultipleLocator(base=0.5)  # this locator puts ticks at regular intervals
            curr_ax.xaxis.set_major_locator(loc)
            curr_ax.xaxis.set_major_formatter(plticker.FuncFormatter(year_formatter))

        # Shrink current axis's height by 10% on the bottom
        # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot
        box = curr_ax.get_position()
        # curr_ax.set_position(
        #     [box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9]
        # )

    fig.suptitle(f"Predictions for {backtests[0].backtest.results[0,benchmark_id].slaw.benchmark}")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=4)
    fig.subplots_adjust(right=0.93)
    cbar_ax = fig.add_axes((0.95, 0.1, 0.01, 0.8))
    fig.colorbar(
        cax=cbar_ax,
        mappable=matplotlib.cm.ScalarMappable(norm=line_color_norm, cmap=color_map),
        location="right",
    )
    # format y ticks
    cbar_ax.yaxis.set_major_formatter(year_formatter)
    plt.show()



@dataclass
class CapabilityBacktest2:
    backtest: BacktestData


def plot_capability_backtesting_figure2(backtests: list[CapabilityBacktest2], benchmark_id: int):
    prediction_title_dict = {
        "PC-1": "PC-1",
        "Elo": "Elo",
        "log10 FLOP_opt": "log-FLOP (1E21)",
        "release_date": "Release Date",
    }

    markerlist = ["o", "x", "s", "D"]

    b0 = backtests[0].backtest
    boundaries = np.array(get_split_boundaries(b0, benchmark_id, padding=0.0))

    color_map = plt.get_cmap("coolwarm")
    line_color_norm = matplotlib.colors.BoundaryNorm(boundaries, color_map.N)
    # shift the boundaries slightly forward so that items on the line are grouped in the previous split
    scatter_color_norm = matplotlib.colors.BoundaryNorm(boundaries + 0.01, color_map.N)

    n_split, n_bench = backtests[0].backtest.results.shape
    assert benchmark_id < n_bench

    fig, ax = plt.subplots(1, len(backtests), figsize=(3 * len(backtests), 3.5), squeeze=False)

    for j, capability_backtest in enumerate(backtests):
        backtest = capability_backtest.backtest
        extrapolatable = False
        _, test_err = compute_test_train_error(backtest.results)

        # first, plot the train points.
        # We need to use the final dataframe to get the ground truth data, since it will have the best-trained PC-1 score (and doesn't matter for the other models)
        bdp_g: BacktestDataPoint[ObsScalingLawPredictor] = backtest.global_split_results[
            benchmark_id
        ]
        bdp_g_copy = bdp_g.copy()
        augment_train_test_slaw(
            bdp_g_copy.slaw,
            bdp_g_copy.model,
            bdp_g_copy.split_train,
            bdp_g_copy.split_test,
        )

        bdp_g_splits = list(backtest.splitter.split(bdp_g_copy.split_train))

        curr_ax = ax[0, j]

        plotted_points = set()
        lines = []
        last_max_v = bdp_g_copy.split_train[backtest.splitter.key].min()
        for i, (train, test) in enumerate(bdp_g_splits):
            max_v = train[backtest.splitter.key].max()
            min_v = last_max_v
            last_max_v = max_v

            df = train[~train[backtest.splitter.key].isin(plotted_points)]

            y_key = f"{bdp_g.slaw.benchmark}"

            true_y = df[y_key]
            pred_y = df[f"{y_key} pred"]


            curr_ax.scatter(
                true_y,
                pred_y,
                label=f"{min_v:.1f} - {max_v:.1f} (Split {i})" if j == 0 else None,
                c=df["release_date"],
                cmap=color_map,
                norm=scatter_color_norm,
                marker=markerlist[i],
                s=20,
            )

            plotted_points.update(train[backtest.splitter.key])

            if i == len(bdp_g_splits) - 1:
                df = bdp_g_copy.split_train[
                    ~bdp_g_copy.split_train[backtest.splitter.key].isin(plotted_points)
                ]
                true_y = df[y_key]
                pred_y = df[f"{y_key} pred"]
                curr_ax.scatter(
                    true_y,
                    pred_y,
                    label=f"{max_v:.1f}+ (Split {i+1})" if j == 0 else None,
                    c=df["release_date"],
                    cmap=color_map,
                    norm=scatter_color_norm,
                    marker=markerlist[i + 1],
                    s=20,
                )

                # plot calibration line through diagonal
                curr_ax.plot(
                    [0.1, 0.7],
                    [0.1, 0.7],
                    color="black",
                    linestyle="--",
                )


                curr_ax.set_title(
                    (
                        prediction_title_dict[bdp_g.model.intermediate]
                        if bdp_g.model.intermediate in prediction_title_dict
                        else bdp_g.model.intermediate
                    ),
                )
                if j == 0:
                    curr_ax.set_ylabel(bdp_g.slaw.benchmark.replace("Raw", "Predicted"))
                curr_ax.set_xlabel(bdp_g.slaw.benchmark.replace(" Raw", ""))


        # Shrink current axis's height by 10% on the bottom
        # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot
        # box = curr_ax.get_position()
        # curr_ax.set_position(
        #     [box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9]
        # )

    fig.suptitle(f"Calibration for {backtests[0].backtest.results[0,benchmark_id].slaw.benchmark}")
    fig.subplots_adjust(right=0.93, top=0.8)
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=4)
    cbar_ax = fig.add_axes((0.95, 0.1, 0.01, 0.8))
    fig.colorbar(
        cax=cbar_ax,
        mappable=matplotlib.cm.ScalarMappable(norm=line_color_norm, cmap=color_map),
        location="right",
    )
    # format y ticks
    cbar_ax.yaxis.set_major_formatter(year_formatter)
    plt.show()


    
def plot_capability_backtesting_figure2_hist(
    backtests: list[CapabilityBacktest2], 
    benchmark_id: int
):
    # Get the test errors for each method
    method_errors = []
    method_names = []
    
    for backtest in backtests:
        _, test_err = compute_test_train_error(backtest.backtest.results)
        method_errors.append(np.sqrt(test_err[:, benchmark_id]))  # Get RMSE for specific benchmark
        method_names.append(backtest.backtest.model_class.__name__.replace("Predictor", ""))

    # Setup the plot
    n_methods = len(backtests)
    n_splits = len(method_errors[0])
    
    # Create positions for bars
    width = 0.15  # Width of each bar
    x = np.arange(n_methods)  # Center positions for each method
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create split labels
    splits = []
    for result in backtests[0].backtest.results.T[0]:
        min_v = result.split_train[backtests[0].backtest.splitter.key].min()
        max_v = result.split_train[backtests[0].backtest.splitter.key].max()
        splits.append(f"{min_v:.1f}-{max_v:.1f}")
    
    # Plot bars for each split
    for i in range(n_splits):
        split_errors = [errors[i] for errors in method_errors]
        offset = (i - n_splits/2 + 0.5) * width
        bars = ax.bar(x + offset, split_errors, width, label=f'Split {splits[i]}')
        ax.bar_label(bars, fmt='%.3f', padding=3, rotation=0)
    
    # Customize the plot
    benchmark_name = backtests[0].backtest.results[0, benchmark_id].slaw.benchmark.replace(" Raw", "")
    ax.set_ylabel('RMSE', size='large')
    ax.set_title(f'Test RMSE by Method for {benchmark_name}', size='x-large')
    
    ax.set_xticks(x)
    ax.set_xticklabels(method_names)
    ax.legend(loc='upper left')
    
    # Add a light grid
    ax.grid(True, axis='y', alpha=0.3)
    
    fig.tight_layout()
    plt.show()

def plot_capability_backtesting_figure3(backtests: list[CapabilityBacktest2], benchmark_id: int):
    prediction_title_dict = {
        "PC-1": "PC-1",
        "Elo": "Elo",
        "log10 FLOP_opt": "log-FLOP (1E21)",
        "release_date": "Release Date",
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
        _, test_err = compute_test_train_error(backtest.results)
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
        _, test_err = compute_test_train_error(backtest.results)

        # Prepare data for the current benchmark
        bdp_g: BacktestDataPoint[ObsScalingLawPredictor] = backtest.global_split_results[benchmark_id]
        bdp_g_copy = bdp_g.copy()
        augment_train_test_slaw(
            bdp_g_copy.slaw,
            bdp_g_copy.model,
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
            prediction_title_dict.get(bdp_g.model.intermediate, bdp_g.model.intermediate)
        )
        if j == 0:
            curr_ax_calib.set_ylabel(bdp_g.slaw.benchmark.replace("Raw", "Predicted"))
        # curr_ax_calib.set_xlabel(bdp_g.slaw.benchmark.replace(" Raw", ""))


        # --- Histogram Plot (Bottom Row) ---
        curr_ax_hist = ax[1, j]
        _, test_err = compute_test_train_error(backtest.results)
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
    
def plot_errmatrix_comparison(
    backtests: list[BacktestData],
):
    assert len(backtests) > 0
    methods = [b.model_class.__name__.replace("Predictor", "") for b in backtests]
    # create 3 graphs for each split in [test, train]:
    # 1. Aggregate over benchmarks
    # 2. Aggregate over splits
    # 3. Aggregate over both
    fig, ax = plt.subplots(2, 3, figsize=(30 * 0.65, 20 * 0.65), squeeze=False)

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
        train_err, test_err = compute_test_train_error(b.results)
        train_errs[i] = train_err
        test_errs[i] = test_err

    train_vmax = np.max(np.sqrt(train_errs)).item()
    test_vmax = np.max(np.sqrt(test_errs)).item()

    benchmarks = [b.replace(" Raw", "") for b in backtests[0].benchmarks]

    # aggregate over splits
    sns.heatmap(
        np.sqrt(train_errs.mean(axis=1).T),
        ax=ax[0, 0],
        yticklabels=benchmarks,
        xticklabels=methods,
        annot=True,
        vmin=0,
        vmax=train_vmax,
    )
    sns.heatmap(
        np.sqrt(test_errs.mean(axis=1).T),
        ax=ax[1, 0],
        yticklabels=benchmarks,
        xticklabels=methods,
        annot=True,
        vmin=0,
        vmax=test_vmax,
    )

    splits = []
    for result in backtests[0].results.T[0]:
        min_v = result.split_train[backtests[0].splitter.key].min()
        max_v = result.split_train[backtests[0].splitter.key].max()
        splits.append(f"{min_v:.1f} - {max_v:.1f}")

    # aggregate over benchmarks
    sns.heatmap(
        np.sqrt(train_errs.mean(axis=2).T),
        ax=ax[0, 1],
        yticklabels=splits,
        xticklabels=methods,
        annot=True,
        vmin=0,
        vmax=train_vmax,
    )
    sns.heatmap(
        np.sqrt(test_errs.mean(axis=2).T),
        ax=ax[1, 1],
        yticklabels=splits,
        xticklabels=methods,
        annot=True,
        vmin=0,
        vmax=test_vmax,
    )

    # aggregate over methods
    sns.heatmap(
        np.sqrt(train_errs.mean(axis=0).T),
        ax=ax[0, 2],
        yticklabels=benchmarks,
        xticklabels=splits,
        annot=True,
        vmin=0,
        vmax=train_vmax,
    )
    sns.heatmap(
        np.sqrt(test_errs.mean(axis=0).T),
        ax=ax[1, 2],
        yticklabels=benchmarks,
        xticklabels=splits,
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
    method: str,
    backtest: BacktestData,
):
    # Create a heatmap with the following rows

    # create 3 graphs for each split in [test, train]:
    # 1. Aggregate over benchmarks
    # 2. Aggregate over splits
    # 3. Aggregate over both
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), squeeze=False)

    _, test_err = compute_test_train_error(backtest.results)
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

    benchmarks = [b.replace(" Raw", "") for b in backtest.benchmarks]

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

    ax[0, 0].hlines(len(backtest.splits), *ax[0, 0].get_xlim(), color="black", linewidth=4)
    ax[0, 0].vlines(len(backtest.benchmarks), *ax[0, 0].get_ylim(), color="black", linewidth=4)
    ax[0, 0].set_xlabel("Benchmarks", size="large")

    # set column titles
    ax[0, 0].set_title(f"{method} Test RMSE", size="x-large")

    fig.tight_layout()
    plt.show()


def plot_all_loss_curves(data: BacktestData):
    n_split, n_bench = data.results.shape
    fig, ax = plt.subplots(
        n_split + 1,
        n_bench,
        figsize=(4 * n_bench, 4 * (n_split + 1)),
        squeeze=False,
    )
    for split_idx in range(n_split + 1):
        for bench_idx in range(n_bench):
            bdp: BacktestDataPoint[ObsScalingLawPredictor] = (
                data.results[split_idx, bench_idx]
                if split_idx < n_split
                else data.global_split_results[bench_idx]
            )
            slaw = bdp.slaw
            ax[split_idx, bench_idx].plot(np.log10(slaw.train_losses[500:]), label="train")
            ax[split_idx, bench_idx].set_title(slaw.benchmark)
            ax[split_idx, bench_idx].legend()
    plt.show()

    fig, ax = plt.subplots(
        n_split + 1,
        n_bench,
        figsize=(4 * n_bench, 4 * (n_split + 1)),
        squeeze=False,
    )
    for split_idx in range(n_split + 1):
        for bench_idx in range(n_bench):
            bdp: BacktestDataPoint[ObsScalingLawPredictor] = (
                data.results[split_idx, bench_idx]
                if split_idx < n_split
                else data.global_split_results[bench_idx]
            )
            model = bdp.model
            slaw = bdp.slaw
            ax[split_idx, bench_idx].plot(np.log10(model.train_losses[:]), label="train")
            ax[split_idx, bench_idx].set_title(slaw.benchmark)
            ax[split_idx, bench_idx].legend()
    plt.show()


def plot_slaw[T: ObsScalingLawPredictor](
    point: BacktestDataPoint[T],
):
    # now plot the data for the actual fit curve on the excluded benchmark
    # 1 row, 4 columns
    # col 0: FLOPs vs benchmark (show both true and predicted)
    # col 1: FLOPs vs logit benchmark (show both true and predicted)
    # col 2: capability vs benchmark (show both true and predicted)
    # col 3: capability vs logit benchmark (show both true and predicted)

    pt = point.copy()
    augment_train_test_slaw(pt.slaw, pt.model, pt.split_train, pt.split_test)

    fig, ax = plt.subplots(1, 4, figsize=(20, 5), squeeze=False)  # 4 columns
    ax_arr = ax[0]
    # plot in flop x-space and benchmark y-space
    plot_train_test(
        ax_arr[0],
        pt.split_train,
        pt.split_test,
        "log10 FLOP_opt",
        [
            Spe(f"{pt.slaw.benchmark}", "Ground Truth", "C0"),
            Spe(f"{pt.slaw.benchmark} pred", "Prediction", "C1"),
        ],
        y_label=pt.slaw.benchmark,
    )

    plot_train_test(
        ax_arr[1],
        pt.split_train,
        pt.split_test,
        "log10 FLOP_opt",
        [
            Spe(f"{pt.slaw.benchmark} logit", "Ground Truth", "C0"),
            Spe(f"{pt.slaw.benchmark} logit pred", "Prediction", "C1"),
        ],
        y_label=f"{pt.slaw.benchmark} logit",
    )

    plot_train_test(
        ax_arr[2],
        pt.split_train,
        pt.split_test,
        "PC-1",
        [
            Spe(f"{pt.slaw.benchmark}", "Ground Truth", "C0"),
            Spe(f"{pt.slaw.benchmark} pred", "Prediction", "C1"),
        ],
        y_label=pt.slaw.benchmark,
    )

    plot_train_test(
        ax_arr[3],
        pt.split_train,
        pt.split_test,
        "PC-1",
        [
            Spe(f"{pt.slaw.benchmark} logit", "Ground Truth", "C0"),
            Spe(f"{pt.slaw.benchmark} logit pred", "Prediction", "C1"),
        ],
        y_label=f"{pt.slaw.benchmark} logit",
    )

    plt.show()


def plot_linear_scaling_law(lin_data_point: BacktestDataPoint[LinearPC1Predictor]):
    fig, ax = plt.subplots(
        len(lin_data_point.model.benchmarks),
        2,
        figsize=(10, len(lin_data_point.model.benchmarks) * 5),
        squeeze=False,
    )  # 1 columns

    # insert data from excluded benchmark

    for bench_idx, benchmark in enumerate(lin_data_point.model.benchmarks):
        pt = lin_data_point.copy()
        augment_train_test_linear(pt.model, pt.split_train, pt.split_test)
        ax_arr = ax[bench_idx]
        benchmark = pt.model.benchmarks[bench_idx]
        plot_train_test(
            ax_arr[0],
            pt.split_train,
            pt.split_test,
            "log10 FLOP_opt",
            [
                Spe(f"{benchmark}", "Ground Truth", "C0"),
                Spe(f"{benchmark} pred", "Prediction", "C1"),
            ],
            y_label=benchmark,
        )
        plot_train_test(
            ax_arr[1],
            pt.split_train,
            pt.split_test,
            "PC-1 (linear)",
            [
                Spe(f"{benchmark}", "Ground Truth", "C0"),
                Spe(f"{benchmark} pred", "Prediction", "C1"),
            ],
            y_label=benchmark,
        )

    plot_slaw(lin_data_point)


def plot_flop_scaling_law(flop_data_point: BacktestDataPoint[DirectLogFlopPredictor]):
    # fig, ax = plt.subplots(
    #     len(algprog_flop_data_point.model.benchmarks),
    #     4,
    #     figsize=(4 * 4, len(algprog_flop_data_point.model.benchmarks) * 4),
    #     squeeze=False,
    # )  # 1 columns

    # for bench_idx, benchmark in enumerate(logit_data_point.model.benchmarks):
    #     pt = logit_data_point.copy()
    #     augment_train_test_logit(pt.model, pt.split_train, pt.split_test)
    #     plot_logit_model(
    #         ax[bench_idx], bench_idx, pt.split_train, pt.split_test, pt.model
    #     )

    # plt.tight_layout()
    # plt.show()

    slaw = flop_data_point.slaw
    print("slaw beta", slaw.beta.item())
    print("slaw alpha", slaw.alpha.item())

    plt.plot(np.log10(slaw.train_losses[500:]), label="slaw")
    plt.show()

    plot_slaw(flop_data_point)


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

ewbs_lin_data = backtest_models_metric(
    ewbs, LinearPC1Predictor, openllm_elo_merged, openllm_elo_benchmarks
)
ewbs_lin_train_err, ewbs_lin_test_err = compute_test_train_error(ewbs_lin_data.results)

# %%
ewbs_elo_data = backtest_models_metric(
    ewbs, DirectEloPredictor, openllm_elo_merged, openllm_elo_benchmarks
)
ewbs_elo_train_err, ewbs_elo_test_err = compute_test_train_error(ewbs_elo_data.results)


# %%
ewbs_flop_data = backtest_models_metric(
    ewbs, DirectLogFlopPredictor, openllm_elo_merged, openllm_elo_benchmarks
)
ewbs_flop_train_err, ewbs_flop_test_err = compute_test_train_error(ewbs_flop_data.results)


# %%
ewbs_date_data = backtest_models_metric(
    ewbs, DirectDatePredictor, openllm_elo_merged, openllm_elo_benchmarks
)
ewbs_date_train_err, ewbs_date_test_err = compute_test_train_error(ewbs_date_data.results)


# %%

split_idx = 2
bench_idx = 3
plot_linear_scaling_law(ewbs_lin_data.results[split_idx, bench_idx])

# %%
plot_linear_scaling_law(ewbs_lin_data.global_split_results[3])


# %%


split_idx = 0
bench_idx = 0
plot_flop_scaling_law(ewbs_flop_data.results[split_idx, bench_idx])

# %%


# print ALL of the average errors:
# Linear, Logit, Algprog, Flop, Elo

print(f"Linear PC1 -> Downstream Train MSE: {ewbs_lin_train_err.mean():.3f}")
print(f"Linear PC1 -> Downstream Test MSE: {ewbs_lin_test_err.mean():.3f}")
print(f"Elo -> Downstream Train MSE: {ewbs_elo_train_err.mean():.3f}")
print(f"Elo -> Downstream Test MSE: {ewbs_elo_test_err.mean():.3f}")
print(f"Flop -> Downstream Train MSE: {ewbs_flop_train_err.mean():.3f}")
print(f"Flop -> Downstream Test MSE: {ewbs_flop_test_err.mean():.3f}")
print(f"Date -> Downstream Train MSE: {ewbs_date_train_err.mean():.3f}")
print(f"Date -> Downstream Test MSE: {ewbs_date_test_err.mean():.3f}")

print()
print()

print(f"Linear PC1 -> Downstream Train RMSE: {ewbs_lin_train_err.mean()**0.5:.3f}")
print(f"Linear PC1 -> Downstream Test RMSE: {ewbs_lin_test_err.mean()**0.5:.3f}")
print(f"Elo -> Downstream Train RMSE: {ewbs_elo_train_err.mean()**0.5:.3f}")
print(f"Elo -> Downstream Test RMSE: {ewbs_elo_test_err.mean()**0.5:.3f}")
print(f"Flop -> Downstream Train RMSE: {ewbs_flop_train_err.mean()**0.5:.3f}")
print(f"Flop -> Downstream Test RMSE: {ewbs_flop_test_err.mean()**0.5:.3f}")
print(f"Date -> Downstream Train RMSE: {ewbs_date_train_err.mean()**0.5:.3f}")
print(f"Date -> Downstream Test RMSE: {ewbs_date_test_err.mean()**0.5:.3f}")

plot_comparison(
    [
        ewbs_lin_data,
        ewbs_elo_data,
        ewbs_flop_data,
        ewbs_date_data,
    ],
    expand=True,
)

# %%
plot_errmatrix_comparison(
    [
        ewbs_lin_data,
        ewbs_elo_data,
        ewbs_flop_data,
        ewbs_date_data,
    ]
)

# %%

plot_all_loss_curves(ewbs_lin_data)


# %%

plot_split(ewbs_flop_data, 0, "log10 FLOP_opt", expand=False, line=True)


# %%
plot_split(ewbs_elo_data, 0, "Elo", expand=False, line=True)


# %%
plot_split(ewbs_lin_data, 0, "PC-1", expand=False, line=True)


# %%
plot_test_errmatrix_single("log-FLOP", ewbs_flop_data)
plot_test_errmatrix_single("Elo", ewbs_elo_data)
plot_test_errmatrix_single("PC-1", ewbs_lin_data)


# %%
plot_capability_backtesting_figure2(
    [
        CapabilityBacktest2(ewbs_lin_data),
        CapabilityBacktest2(ewbs_elo_data),
        CapabilityBacktest2(ewbs_flop_data),
        CapabilityBacktest2(ewbs_date_data),
    ],
    5,
)

plot_capability_backtesting_figure2_hist(
        [
        CapabilityBacktest2(ewbs_lin_data),
        CapabilityBacktest2(ewbs_elo_data),
        CapabilityBacktest2(ewbs_flop_data),
        CapabilityBacktest2(ewbs_date_data),
    ],
    5,
)

#%%
plot_capability_backtesting_figure3(
    [
        CapabilityBacktest2(ewbs_lin_data),
        CapabilityBacktest2(ewbs_elo_data),
        CapabilityBacktest2(ewbs_flop_data),
        CapabilityBacktest2(ewbs_date_data),
    ],
    5,
)

# %%
plot_capability_backtesting_figure(
    [
        CapabilityBacktest(ewbs_lin_data, False),
        CapabilityBacktest(ewbs_elo_data),
        CapabilityBacktest(ewbs_flop_data),
        CapabilityBacktest(ewbs_date_data),
    ],
    5,
)

# %%
