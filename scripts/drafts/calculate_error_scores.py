import pandas as pd
import numpy as np
import xarray as xr
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from typing import Dict, DefaultDict, Tuple, List

from scripts.drafts.gauge_name_lookup import gauge_name_lookup
from src.analysis.evaluation import (
    spatial_rmse,
    spatial_r2,
    spatial_nse,
    spatial_bias,
    spatial_kge,
)
from src.analysis.evaluation import temporal_rmse, temporal_r2, temporal_nse
from src.analysis.evaluation import (
    _nse_func,
    _rmse_func,
    _r2_func,
    _bias_func,
    _kge_func,
    _mse_func,
)
from collections import defaultdict


def error_func(preds_xr: xr.Dataset, error_str: str) -> pd.DataFrame:
    lookup = {
        "nse": _nse_func,
        "mse": _mse_func,
        "kge": _kge_func,
        "bias": _bias_func,
    }
    error_func = lookup[error_str]

    df = preds_xr.to_dataframe()
    df = df.dropna(how="any")
    df = df.reset_index().set_index("time")

    station_ids = df["station_id"].unique()
    errors = []
    for station_id in station_ids:
        d = df.loc[df["station_id"] == station_id]
        if error_str == "rmse":
            _error_calc = error_func(
                d["obs"].values, d["sim"].values, n_instances=d.size
            )
        else:
            _error_calc = error_func(d["obs"].values, d["sim"].values)
        errors.append(_error_calc)

    error = pd.DataFrame({"station_id": station_ids, error_str: errors})

    return error


def calculate_errors(preds: xr.Dataset) -> pd.DataFrame:
    errors = [
        error_func(preds, "nse").set_index("station_id"),
        error_func(preds, "kge").set_index("station_id"),
        error_func(preds, "mse").set_index("station_id"),
        error_func(preds, "bias").set_index("station_id"),
    ]
    error_df = errors[0].join(errors[1].join(errors[2]).join(errors[3])).reset_index()

    return error_df


class FuseErrors:
    def __init__(self, fuse_data: xr.Dataset):
        assert all(
            np.isin(
                [
                    "obs",
                    "SimQ_TOPMODEL",
                    "SimQ_PRMS",
                    "SimQ_ARNOVIC",
                    "SimQ_SACRAMENTO",
                ],
                [v for v in fuse_data.data_vars],
            )
        )

        self.fuse_data = fuse_data
        self._separate_into_das()

        nse_df = self._calculate_metric("nse").drop("Name", axis=1, level=1)
        kge_df = self._calculate_metric("kge").drop("Name", axis=1, level=1)
        bias_df = self._calculate_metric("bias").drop("Name", axis=1, level=1)
        rmse_df = self._calculate_metric("rmse").drop("Name", axis=1, level=1)

        #  convert into one clean dataframe
        fuse_errors = pd.concat([nse_df, rmse_df, kge_df, bias_df], axis=1)
        fuse_errors = self.tidy_dataframe(fuse_errors)
        self.fuse_errors = fuse_errors

    @staticmethod
    def tidy_dataframe(fuse_errors: pd.DataFrame) -> pd.DataFrame:
        try:
            fuse_errors = (
                fuse_errors.drop("time", axis=1, level=1)
                .swaplevel(axis=1)
                .sort_index(axis=1)
            )
        except KeyError:
            pass

        fuse_errors = fuse_errors.rename(
            {"NSE": "nse", "BIAS": "bias", "MSE": "mse"}, axis=1, level=0
        )
        #  Remove the multiple "Name" columns ...
        station_names = pd.DataFrame(gauge_name_lookup, index=["gauge_name"]).T
        fuse_errors["Name"] = station_names
        #  sort the ordering of the multi-index
        fuse_errors = fuse_errors.swaplevel(axis=1).sort_index(axis=1)
        return fuse_errors

    def _separate_into_das(self) -> None:
        #  separate into DataArrays
        self.obs = self.fuse_data["obs"].transpose("station_id", "time")
        topmodel = self.fuse_data["SimQ_TOPMODEL"]
        arnovic = self.fuse_data["SimQ_ARNOVIC"]
        prms = self.fuse_data["SimQ_PRMS"]
        sacramento = self.fuse_data["SimQ_SACRAMENTO"]

        self.model_preds = [topmodel, arnovic, prms, sacramento]
        self.model_names = ["TOPMODEL", "VIC", "PRMS", "Sacramento"]

    def _calculate_metric(self, metric: str) -> None:
        metric_lookup = {
            "nse": spatial_nse,
            "rmse": spatial_rmse,
            "bias": spatial_bias,
            "kge": spatial_kge,
        }
        function = metric_lookup[metric]

        out_list = []
        for model, model_name in tqdm(
            zip(self.model_preds, self.model_names), desc=metric
        ):
            out_list.append(function(self.obs, model).rename(model_name))

        # merge all of the station error metrics into one xr.Dataset
        metric_xr = xr.merge([out_list[0], out_list[1], out_list[2], out_list[3],])
        metric_df = metric_xr.to_dataframe()
        metric_df = (
            pd.DataFrame(gauge_name_lookup, index=["gauge_name"])
            .T.join(metric_df)
            .rename(columns=dict(gauge_name="Name"))
        )
        metric_df.columns = [
            [metric for _ in range(len(metric_df.columns))],
            metric_df.columns,
        ]

        return metric_df

    def get_metric_df(self, metric: str) -> pd.DataFrame:
        #  select only that metric
        df = self.fuse_errors.loc[
            :, self.fuse_errors.columns.get_level_values(0) == metric.lower()
        ].droplevel(level=0, axis=1)
        if not "Name" in df.columns:
            df["Name"] = pd.DataFrame(gauge_name_lookup, index=["gauge_name"]).T

        return df

    def get_model_df(self, model: str) -> pd.DataFrame:
        acceptable_models = [
            n
            for n in np.unique(self.fuse_errors.droplevel(axis=1, level=0).columns)
            if n != "Name"
        ]
        assert model in acceptable_models, f"Require one of: {acceptable_models}"
        df = self.fuse_errors.loc[
            :, self.fuse_errors.columns.get_level_values(1) == model
        ].droplevel(level=1, axis=1)

        if not "Name" in df.columns:
            df["Name"] = pd.DataFrame(gauge_name_lookup, index=["gauge_name"]).T

        return df


class FUSEPublishedScores:
    def __init__(self, fuse_dir: Path):
        assert fuse_dir.exists(), f"Expected {fuse_dir} to exist"
        self.fuse_dir = fuse_dir

    @staticmethod
    def fix_name(string: str):
        string = string.replace("_060", "_TOPMODEL")
        string = string.replace("_230", "_VIC")
        string = string.replace("_342", "_PRMS")
        string = string.replace("_426", "_SACRAMENTO")
        return string

    def read_nse_scores(self) -> pd.DataFrame:
        df = pd.read_csv(
            self.fuse_dir / "Summary_Scores/NSE_decomposed_scores.txt", skiprows=4
        )

        # fix the column names
        df.columns = [self.fix_name(c) for c in df.columns]

        #  rename Gauge_ID
        df = df.rename({"Gauge_ID": "station_id"}, axis=1)

        return df

    def read_best_scores(self) -> pd.DataFrame:
        """[summary]

        Args:
            fuse_dir (Path): [description]

        Note:
        - the Published performance scores are calculated for the period 1993-2008
        - the `Best_Scores.txt` contains the best overall scores from all simuations,
            and so the best score for bias will not relate to the same simulation
            as the best score for NSE.
        """

        df = pd.read_csv(self.fuse_dir / "Summary_Scores/Best_Scores.txt", skiprows=4)

        # fix the column names
        df.columns = [self.fix_name(c) for c in df.columns]

        #  rename Gauge_ID
        df = df.rename({"Gauge_ID": "station_id"}, axis=1)

        return df

    @staticmethod
    def get_metric_from_df(df: pd.DataFrame, metric: str) -> pd.DataFrame:
        acceptable_metrics = np.unique(
            ["_".join(c.split("_")[:-1]) for c in df.columns]
        )
        assert (
            metric in acceptable_metrics
        ), f"Require one of these metrics: {acceptable_metrics}"
        df = df.loc[:, [("id" in c) or (metric in c) for c in df.columns]].set_index(
            "station_id"
        )
        df = df.join(pd.DataFrame(gauge_name_lookup, index=["Name"]).T)

        df.columns = [
            ["NSE" for _ in range(len(df.columns))],
            [c.replace("NSE_", "") for c in df.columns],
        ]
        return df


class DeltaError:
    def __init__(self, ealstm_preds, lstm_preds, fuse_data):
        self._join_into_one_ds(ealstm_preds, lstm_preds, fuse_data)

    def _join_into_one_ds(self, ealstm_preds, lstm_preds, fuse_data) -> None:
        self.all_preds = xr.combine_by_coords(
            [
                ealstm_preds.rename({"sim": "EALSTM"}).drop("obs"),
                lstm_preds.rename({"sim": "LSTM"}),
                (
                    fuse_data.rename(
                        dict(
                            zip(
                                [v for v in fuse_data.data_vars],
                                [
                                    str(v).replace("SimQ_", "")
                                    for v in fuse_data.data_vars
                                ],
                            )
                        )
                    ).drop("obs")
                ),
            ]
        )

    @staticmethod
    def calculate_all_errors(all_preds: xr.DataArray) -> Dict[str, pd.DataFrame]:
        station_names = pd.DataFrame(gauge_name_lookup, index=["gauge_name"]).T
        metrics = ["nse", "kge", "mse", "bias"]

        output_dict = defaultdict(list)
        station_names = pd.DataFrame(gauge_name_lookup, index=["gauge_name"]).T

        # Calculate Model Error Metrics for each model
        output_dict = defaultdict(list)
        station_names = pd.DataFrame(gauge_name_lookup, index=["gauge_name"]).T
        for ix, model in tqdm(
            enumerate([v for v in all_preds.data_vars if v != "obs"])
        ):
            _errors = calculate_errors(
                all_preds[["obs", model]].rename({model: "sim"})
            ).set_index("station_id")

            for metric in ["nse", "kge", "mse", "bias"]:
                output_dict[metric].append(
                    _errors.rename({metric: model}, axis=1)[model]
                )

        # merge into single dataframe
        errors_dict = {}
        for metric in metrics:
            errors_dict[metric] = pd.concat(output_dict[metric], axis=1)

        return errors_dict

    @staticmethod
    def calculate_error_diff(error_df: pd.DataFrame, ref_model: str = "LSTM") -> pd.DataFrame:
        all_deltas = []
        for model in [c for c in error_df.columns if c != ref_model]:
            delta = error_df[ref_model] - error_df[model]
            delta.name = model
            all_deltas.append(delta)

        delta_df = pd.concat(all_deltas, axis=1)

        return delta_df

    def calculate_all_delta_dfs(self, errors_dict: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame]]:
        lstm_delta: Dict[str, pd.DataFrame] = defaultdict()
        ealstm_delta: Dict[str, pd.DataFrame] = defaultdict()

        for metric in [k for k in errors_dict.keys()]:
            lstm_delta[metric] = self.calculate_error_diff(error_df=errors_dict[metric], ref_model="LSTM")
            ealstm_delta[metric] = self.calculate_error_diff(error_df=errors_dict[metric], ref_model="EALSTM")

        return lstm_delta, ealstm_delta
