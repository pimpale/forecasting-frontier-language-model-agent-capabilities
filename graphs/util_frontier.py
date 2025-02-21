from typing import Optional, override
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn as nn
from util_obs_scaling_law_predictor import ScalingLaw


def get_running_top_n(
    df: pd.DataFrame, x_column: str, y_column: str, n: int, id_column: str
) -> pd.DataFrame:
    """
    This function returns all models that are in the top n of y_column at any point in time, where time is given by x_column.
    """
    top_ids = set()
    x_values = df[x_column].unique()

    for x in x_values:
        data_until_x = df[df[x_column] <= x]
        top_n_at_date = data_until_x.nlargest(n, y_column)[id_column]
        top_ids.update(top_n_at_date)

    return df[df[id_column].isin(top_ids)]


def get_running_top_n_2d(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    z_column: str,
    n: int,
    id_column: str,
) -> pd.DataFrame:
    """
    This function returns all models that are in the top n of z_column for any x,y pair in xy_columns.
    """
    top_ids = set()
    xy_values = df[[x_column, y_column]].drop_duplicates().values

    for x, y in xy_values:
        data_until_xy = df[(df[x_column] <= x) & (df[y_column] <= y)]
        top_n_at_date = data_until_xy.nlargest(n, z_column)[id_column]
        top_ids.update(top_n_at_date)

    return df[df[id_column].isin(top_ids)]


def vectorized_highest_score(
    df, x_column: str, x_column_thresholds: np.ndarray, key: str
):
    """
    Vectorized function to return the highest `key` score for each threshold.

    Parameters:
    df (pd.DataFrame): The dataframe to search.
    x_column (str): The column to threshold using x_column_thresholds.
    x_column_thresholds (np.ndarray): Array of thresholds.
    key (str): The key to search for the highest score.

    Returns:
    np.ndarray: Array of highest `key` scores.
    """
    # Create an array to store the highest scores
    highest_scores = np.zeros(len(x_column_thresholds))

    for i, x in enumerate(x_column_thresholds):
        mask = df[x_column] <= x
        if mask.any():
            highest_scores[i] = df.loc[mask, key].max()
        else:
            highest_scores[i] = np.nan  # or some other placeholder for no data

    return highest_scores


def vectorized_highest_score_df(
    df: pd.DataFrame,
    x_column: str,
    x_column_thresholds: npt.NDArray[np.float32],
    key: str,
) -> pd.DataFrame:
    """
    Vectorized function to return the highest `key` score for each threshold.

    Parameters:
    df (pd.DataFrame): The dataframe to search.
    x_column (str): The column to threshold using x_column_thresholds.
    x_column_thresholds (np.ndarray): Array of numerical values.
    key (str): The key to search for the highest score.

    Returns:
    pd.DataFrame: Returns a dataframe with as many rows as x_column_thresholds. Each row has an x_column value equal to the threshold and the other values are taken from the highest element in df that is below the threshold with regard to `key`
    """
    if df.empty:
        result = pd.DataFrame({x_column: x_column_thresholds})
        for col in df.columns:
            if col != x_column:
                result[col] = np.nan
        return result

    sorted_df = df.sort_values(x_column).reset_index(drop=True)
    sorted_df["cummax_key"] = sorted_df[key].cummax()
    mask = sorted_df[key] == sorted_df["cummax_key"]
    filtered_df = sorted_df[mask].copy()

    thresholds_df = pd.DataFrame(
        {
            x_column: x_column_thresholds,
            "_original_order": np.arange(len(x_column_thresholds)),
        }
    )
    thresholds_df_sorted = thresholds_df.sort_values(x_column)

    merged = pd.merge_asof(
        thresholds_df_sorted, filtered_df, on=x_column, direction="backward"
    )

    merged = merged.sort_values("_original_order").drop(columns=["_original_order"])
    if "cummax_key" in merged.columns:
        merged.drop(columns=["cummax_key"], inplace=True)

    return merged


def vectorized_highest_score_2d(
    df, x_column, x_column_thresholds, y_column, y_column_thresholds, key
):
    """
    Vectorized function to return the highest `key` score for each combination of x_column and y_column.

    Parameters:
    df (pd.DataFrame): The dataframe to search.
    x_column (str): The column to threshold using x_column_thresholds.
    x_column_thresholds (np.ndarray): Array of thresholds
    y_column (str): The column to threshold using y_column_thresholds.
    y_column_thresholds (np.ndarray): Array of thresholds.
    key (str): The key to search for the highest score.

    Returns:
    np.ndarray: 2D array of highest `key` scores.
    """
    # Create a 2D array to store the highest scores
    highest_scores = np.zeros((len(x_column_thresholds), len(y_column_thresholds)))

    for i, x_threshold in enumerate(x_column_thresholds):
        mask = df[x_column] <= x_threshold
        for j, y_threshold in enumerate(y_column_thresholds):
            combined_mask = mask & (df[y_column] <= y_threshold)
            if combined_mask.any():
                highest_scores[i, j] = df.loc[combined_mask, key].max()
            else:
                highest_scores[i, j] = np.nan  # or some other placeholder for no data

    return highest_scores


class Frontier(nn.Module):
    """
    Parent class for all frontier predictors.
    Basically just a linear model that predicts a capability score from the given tensor
    """

    def __init__(
        self,
        # in shape (*, n_samples, n_benchmarks)
        benchmark_data: torch.Tensor,
        # in shape (*, n_samples)
        capability_data: torch.Tensor,
    ):
        super().__init__()
        self.train_benchmark_data = benchmark_data
        self.train_capability_data = capability_data

        # compute linear fit
        # shape: (*,  n_samples, n_benchmarks+1)
        X_design = torch.cat(
            [
                # shape: (*, n_samples, n_benchmarks)
                benchmark_data,
                # shape: (*, n_samples, 1)
                torch.ones_like(capability_data).unsqueeze(-1),
            ],
            dim=-1,
        ).to(torch.float64)

        # shape: (*, n_samples)
        y = capability_data.to(torch.float64)

        self.beta = torch.linalg.lstsq(
            X_design, y, rcond=1e-12, driver="gelsd"
        ).solution.to(torch.float32)

    @classmethod
    def uses_benchmark(cls, s: str) -> bool:
        """
        Return whether the predictor uses the benchmark with the given name.
        """
        return any(s in b.split("*") for b in cls.necessary_benchmarks())

    @staticmethod
    def capability() -> str:
        """
        Return the capability that this predictor predicts.
        """
        raise NotImplementedError

    @staticmethod
    def necessary_benchmarks() -> list[str]:
        """
        Return the list of benchmarks that are necessary for this predictor.
        These benchmarks must appear first in the tensor that is passed into the constructor,
        and the order of the benchmarks must be the same as the order of the benchmarks in this list.
        """
        raise NotImplementedError

    def predict_frontier_capability_scores(
        self,
        # in (*, n_samples, n_benchmarks)
        test_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict capability scores from test data.
        """
        # shape: (*, n_samples, n_benchmarks+1)
        X = torch.cat(
            [
                test_scores,
                torch.ones_like(test_scores[..., 0]).unsqueeze(-1),
            ],
            dim=-1,
        )

        # shape: (*, n_samples)
        return (X @ self.beta.unsqueeze(-1)).squeeze(-1)


class FrontierFlopPredictor(Frontier):
    @override
    @staticmethod
    def capability() -> str:
        return "log10 FLOP_opt"

    @override
    @staticmethod
    def necessary_benchmarks() -> list[str]:
        return ["log10 FLOP_opt"]


class FrontierDatePredictor(Frontier):
    @override
    @staticmethod
    def capability() -> str:
        return "release_date"

    @override
    @staticmethod
    def necessary_benchmarks() -> list[str]:
        return ["release_date"]


class FrontierFlopToEloPredictor(Frontier):
    @override
    @staticmethod
    def capability() -> str:
        return "Elo"

    @override
    @staticmethod
    def necessary_benchmarks() -> list[str]:
        return ["log10 FLOP_opt"]


class FrontierDateToEloPredictor(Frontier):
    @override
    @staticmethod
    def capability() -> str:
        return "Elo"

    @override
    @staticmethod
    def necessary_benchmarks() -> list[str]:
        return ["release_date"]


class FrontierDateAllElicitedToEloPredictor(Frontier):
    @override
    @staticmethod
    def capability() -> str:
        return "Elo"

    @override
    @staticmethod
    def necessary_benchmarks() -> list[str]:
        return ["release_date", "elicited_all_frontier"]


class FrontierDateBestElicitedToEloPredictor(Frontier):
    @override
    @staticmethod
    def capability() -> str:
        return "Elo"

    @override
    @staticmethod
    def necessary_benchmarks() -> list[str]:
        return ["release_date", "elicited_best"]


class FrontierDateDistinctElicitedToEloPredictor(Frontier):
    def __init__(
        self,
        # in shape (*, n_samples, 2)
        benchmark_data: torch.Tensor,
        # in shape (*, n_samples)
        capability_data: torch.Tensor,
    ):
        super().__init__(
            benchmark_data,
            capability_data,
        )

        # compute linear fit
        # shape: (*,  n_samples, 4)
        X_design = torch.stack(
            [
                # shape: (*, n_samples)
                benchmark_data[..., 0] * (1 - benchmark_data[..., -1]),
                # shape: (*, n_samples)
                benchmark_data[..., 0] * benchmark_data[..., -1],
                # shape: (*, n_samples)
                torch.ones_like(capability_data) * (1 - benchmark_data[..., -1]),
                # shape: (*, n_samples)
                torch.ones_like(capability_data) * benchmark_data[..., -1],
            ],
            dim=-1,
        ).to(torch.float64)
        # shape: (*, n_samples)
        y = capability_data.to(torch.float64)

        self.beta = torch.linalg.lstsq(
            X_design, y, rcond=1e-12, driver="gelsd"
        ).solution.to(torch.float32)

    @override
    def predict_frontier_capability_scores(
        self,
        # in (*, n_samples, n_benchmarks)
        test_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict capability scores from test data.
        """

        # shape: (*, n_samples, n_benchmarks+2)
        X = torch.stack(
            [
                test_scores[..., 0] * (1 - test_scores[..., -1]),
                test_scores[..., 0] * test_scores[..., -1],
                torch.ones_like(test_scores[..., 0]) * (1 - test_scores[..., -1]),
                torch.ones_like(test_scores[..., 0]) * test_scores[..., -1],
            ],
            dim=-1,
        )

        # shape: (*, n_samples)
        return (X @ self.beta.unsqueeze(-1)).squeeze(-1)

    @override
    @staticmethod
    def capability() -> str:
        return "Elo"

    @override
    @staticmethod
    def necessary_benchmarks() -> list[str]:
        return ["release_date", "elicited_distinct"]


class FrontierFlopToPC1Predictor(Frontier):
    @override
    @staticmethod
    def capability() -> str:
        return "PC1"

    @override
    @staticmethod
    def necessary_benchmarks() -> list[str]:
        return ["log10 FLOP_opt"]


class FrontierDateToPC1Predictor(Frontier):
    @override
    @staticmethod
    def capability() -> str:
        return "PC1"

    @override
    @staticmethod
    def necessary_benchmarks() -> list[str]:
        return ["release_date"]


class FrontierFlopDateToEloPredictor(Frontier):
    @override
    @staticmethod
    def capability() -> str:
        return "Elo"

    @override
    @staticmethod
    def necessary_benchmarks() -> list[str]:
        return ["release_date", "log10 FLOP_opt"]


class FrontierFlopDateToPC1Predictor(Frontier):
    @override
    @staticmethod
    def capability() -> str:
        return "PC1"

    @override
    @staticmethod
    def necessary_benchmarks() -> list[str]:
        return ["release_date", "log10 FLOP_opt"]
