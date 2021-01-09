import pandas as pd
import numpy as np
import xarray as xr
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from typing import Dict, DefaultDict, Tuple, List, Optional
from HydroErr import HydroErr as he

from scripts.drafts.gauge_name_lookup import gauge_name_lookup
from src.analysis.evaluation import (
    spatial_rmse,
    spatial_r2,
    spatial_nse,
    spatial_bias,
    spatial_pbias,
    spatial_kge,
    spatial_abs_pct_bias,
    spatial_mape,
)
from src.analysis.evaluation import temporal_rmse, temporal_r2, temporal_nse
from src.analysis.evaluation import (
    _nse_func,
    _rmse_func,
    _r2_func,
    _bias_func,
    _pbias_func,
    _kge_func,
    _mse_func,
    _abs_pct_bias_func,
    _mape_func,
    _relative_bias_func,
    _variability_ratio_func,
)
from collections import defaultdict
import sys

sys.path.append("/home/tommy/neuralhydrology")
from neuralhydrology.evaluation.metrics import calculate_all_metrics, calculate_metrics

sys.path.append("/home/tommy/ml_drought")
from src.utils import create_shape_aligned_climatology


def assign_wateryear(dt):
    """https://stackoverflow.com/a/52615358/9940782"""
    dt = pd.Timestamp(dt)
    if dt.month >= 10:
        return pd.datetime(dt.year + 1, 1, 1).year
    else:
        return pd.datetime(dt.year, 1, 1).year


def xr_mam30_ape(preds: xr.Dataset) -> xr.Dataset:
    assert "time" in preds.coords
    # calculate the 30d moving average (30dMA)
    move_avg_30 = preds.rolling(time=30).mean()
    #  calculate the mean annual minumum (MAM) = mean(minimum 30dMA for each water year)
    move_avg_30 = move_avg_30.assign_coords(
        wy=("time", [assign_wateryear(dt) for dt in move_avg_30.time.values])
    )
    mam_30 = (
        move_avg_30.groupby("wy").min(dim="time").isel(wy=slice(1, -1)).mean(dim="wy")
    )

    # calculate the absolute percentage error for MAM 30day
    return np.abs(((mam_30["obs"] - mam_30["sim"]) / mam_30["obs"])) * 100


def error_func(
    preds_xr: xr.Dataset, error_str: str, epsilon: float = 1e-10
) -> pd.DataFrame:
    lookup = {
        "nse": _nse_func,
        "mse": _mse_func,
        "kge": _kge_func,
        "bias": _bias_func,
        "bias_error": _relative_bias_func,
        "std_error": _variability_ratio_func,
        "pbias": _pbias_func,
        "log_nse": _nse_func,
        "inv_kge": _kge_func,
        "sqrt_kge": _kge_func,
        "abs_pct_bias": _abs_pct_bias_func,
        "mape": _mape_func,
    }
    error_func = lookup[error_str]

    df = preds_xr.to_dataframe()

    #  Remove nans and inf values (using the HydroError Package)
    # TODO: ENSURE THIS IS BEFORE INV/LOG
    # sim, obs = he.treat_values(df.sim, df.obs,
    #                            replace_nan=None,
    #                            replace_inf=None,
    #                            remove_neg=True,
    #                            remove_zero=False
    # )
    # df["obs"] = obs
    # df["sim"] = sim

    df = df.dropna(how="any")
    df = df.reset_index().set_index("time")

    station_ids = df["station_id"].unique()
    metrics = []
    for station_id in station_ids:
        d = df.loc[df["station_id"] == station_id]

        try:
            if "log" in error_str:
                _error_calc = error_func(
                    np.log(d["obs"].values + epsilon), np.log(d["sim"].values + epsilon)
                )
            elif "inv" in error_str:
                _error_calc = error_func(
                    (1 / (d["obs"].values + epsilon)), (1 / (d["sim"].values + epsilon))
                )
            elif "sqrt" in error_str:
                _error_calc = error_func(
                    np.sqrt(d["obs"].values), np.sqrt(d["sim"].values)
                )

            else:
                _error_calc = error_func(d["obs"].values, d["sim"].values)
        except RuntimeError:
            _error_calc = np.nan
        metrics.append(_error_calc)

    error = pd.DataFrame({"station_id": station_ids, error_str: metrics})

    return error


def kge_decomposition(
    preds: xr.Dataset, transformation: Optional[str] = None, epsilon: float = 1e-10
) -> pd.DataFrame:
    df = preds.to_dataframe()
    df = df.dropna(how="any")
    df = df.reset_index().set_index("time")

    station_ids = df["station_id"].unique()
    correlations = []
    bias_ratios = []
    variability_ratios = []
    for station_id in station_ids:
        d = df.loc[df["station_id"] == station_id]
        # extract the discharge values
        if transformation == "inverse":
            true_vals = (1 / (d["obs"].values + epsilon))
            pred_vals = (1 / (d["sim"].values + epsilon))
            correlation_str = "inv_correlation"
            bias_ratio_str = "inv_bias_ratio"
            variability_ratio_str = "inv_variability_ratio"

        elif transformation == "sqrt":
            true_vals = np.sqrt(d["obs"].values)
            pred_vals = np.sqrt(d["sim"].values)
            correlation_str = "sqrt_correlation"
            bias_ratio_str = "sqrt_bias_ratio"
            variability_ratio_str = "sqrt_variability_ratio"

        elif transformation is None:
            true_vals = d["obs"].values
            pred_vals = d["sim"].values
            correlation_str = "correlation"
            bias_ratio_str = "bias_ratio"
            variability_ratio_str = "variability_ratio"

        # calculate the decomposed kge components
        r, beta, gamma = _kge_func(true_vals, pred_vals, decomposed_results=True)

        correlations.append(r)
        bias_ratios.append(beta)
        variability_ratios.append(gamma)

    error = pd.DataFrame(
        {
            "station_id": station_ids,
            correlation_str: correlations,
            bias_ratio_str: bias_ratios,
            variability_ratio_str: variability_ratios,
        }
    )
    return error


def calculate_errors(preds: xr.Dataset) -> pd.DataFrame:

    error_mam30 = xr_mam30_ape(preds).to_dataframe("mam30_ape")
    errors = [
        error_func(preds, "nse").set_index("station_id"),
        error_func(preds, "kge").set_index("station_id"),
        error_func(preds, "mse").set_index("station_id"),
        error_func(preds, "bias").set_index("station_id"),
        error_func(preds, "pbias").set_index("station_id"),
        error_func(preds, "log_nse").set_index("station_id"),
        error_func(preds, "inv_kge").set_index("station_id"),
        error_func(preds, "sqrt_kge").set_index("station_id"),
        error_func(preds, "abs_pct_bias").set_index("station_id"),
        error_func(preds, "mape").set_index("station_id"),
        error_func(preds, "bias_error").set_index("station_id"),
        error_func(preds, "std_error").set_index("station_id"),
    ]
    error_df = (
        errors[0]
        .join(errors[1])
        .join(errors[2])
        .join(errors[3])
        .join(errors[4])
        .join(errors[5])
        .join(errors[6])
        .join(errors[7])
        .join(errors[8])
        .join(errors[9])
        .join(errors[10])
        .join(errors[11])
        .join(error_mam30)
        .reset_index()
    )

    return error_df


def calculate_all_data_errors(sim_obs_data: xr.Dataset, decompose_kge: bool = False):
    assert all(np.isin(["obs"], list(sim_obs_data.data_vars)))
    model_var_list: List[str] = [v for v in sim_obs_data.data_vars if "obs" not in v]

    output_dict = defaultdict(dict)
    for model in tqdm(model_var_list, desc="Errors"):
        preds = sim_obs_data[["obs", model]].rename({model: "sim"})
        error_df = calculate_errors(preds).set_index("station_id")
        error_df["rmse"] = np.sqrt(error_df["mse"])

        if decompose_kge:
            decompose_df = kge_decomposition(preds).set_index("station_id")
            error_df = error_df.join(decompose_df)
            inv_decompose_df = kge_decomposition(preds, transformation="inverse").set_index(
                "station_id"
            )
            error_df = error_df.join(inv_decompose_df)
            sqrt_decompose_df = kge_decomposition(preds, transformation="sqrt").set_index(
                "station_id"
            )
            error_df = error_df.join(sqrt_decompose_df)

        output_dict[model.replace("SimQ_", "")] = error_df

    return output_dict


def get_metric_dataframes_from_output_dict(
    output_dict: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    models = list(output_dict.keys())
    metrics = [c for c in output_dict[models[0]].columns if "station_id" != c]
    if "station_id" in output_dict[models[0]].columns:
        index = output_dict[models[0]]["station_id"]
    else:
        index = output_dict[models[0]].index

    all_stations = []
    for model in output_dict.keys():
        all_stations.append(output_dict[model].index)

    all_stations = list(set(all_stations[0]).intersection(*all_stations[:1]))

    metric_dict = {}
    for metric in metrics:
        df_dict = {}
        for model in models:
            df_dict[model] = output_dict[model].loc[all_stations, metric].values
        d = pd.DataFrame(df_dict, index=index)

        metric_dict[metric] = d

    return metric_dict


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
        pbias_df = self._calculate_metric("pbias").drop("Name", axis=1, level=1)
        rmse_df = self._calculate_metric("rmse").drop("Name", axis=1, level=1)
        lognse_df = self._calculate_metric("log_nse").drop("Name", axis=1, level=1)
        invkge_df = self._calculate_metric("inv_kge").drop("Name", axis=1, level=1)
        mape_df = self._calculate_metric("mape").drop("Name", axis=1, level=1)
        abs_pct_bias_df = self._calculate_metric("abs_pct_bias").drop(
            "Name", axis=1, level=1
        )

        #  convert into one clean dataframe
        fuse_errors = pd.concat(
            [nse_df, kge_df, bias_df, pbias_df, lognse_df, invkge_df, mape_df, abs_pct_bias_df],
            axis=1,
        )
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

    def _calculate_metric(self, metric: str, epsilon: float = 1e-10) -> None:
        metric_lookup = {
            "nse": spatial_nse,
            "rmse": spatial_rmse,
            "bias": spatial_bias,
            "pbias": spatial_pbias,
            "kge": spatial_kge,
            "log_nse": spatial_nse,
            "inv_kge": spatial_kge,
            "abs_pct_bias": spatial_abs_pct_bias,
            "mape": spatial_mape,
        }
        function = metric_lookup[metric]

        out_list = []
        for model, model_name in tqdm(
            zip(self.model_preds, self.model_names), desc=metric
        ):
            if "log" in metric:
                obs_copy = np.log(self.obs + epsilon)
                model = np.log(model + epsilon)
            elif "inv" in metric:
                obs_copy = 1 / self.obs + epsilon
                model = 1 / model + epsilon
            else:
                obs_copy = self.obs.copy()

            out_list.append(function(obs_copy, model).rename(model_name))

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
        #  select only that metric (error catching - level 0 or level 1)
        df = self.fuse_errors.loc[
            :, self.fuse_errors.columns.get_level_values(0) == metric.lower()
        ].droplevel(level=0, axis=1)
        if df.shape[-1] == 0:
            df = self.fuse_errors.loc[
                :, self.fuse_errors.columns.get_level_values(1) == metric.lower()
            ].droplevel(level=1, axis=1)

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
    def __init__(
        self,
        ealstm_preds,
        lstm_preds,
        fuse_data,
        benchmark_calculation_ds: Optional[xr.Dataset] = None,
        incl_benchmarks: bool = True,
    ):
        if incl_benchmarks:
            assert (
                benchmark_calculation_ds is not None
            ), "Must provide if incl_benchmarks"
            assert "discharge_spec" in list(benchmark_calculation_ds.data_vars)
        self.all_preds = self._join_into_one_ds(ealstm_preds, lstm_preds, fuse_data)
        if incl_benchmarks:
            self.all_preds = self.calculate_benchmarks(benchmark_calculation_ds)

    def calculate_benchmarks(self, benchmark_calculation_ds: xr.Dataset):
        all_preds = self.all_preds
        # 1) Persistence
        all_preds["persistence"] = (
            benchmark_calculation_ds["discharge_spec"]
            .shift(time=1)
            .sel(station_id=all_preds.station_id, time=all_preds.time)
        )

        #  2) DayofYear Climatology
        climatology_unit = "month"

        climatology_doy = (
            benchmark_calculation_ds["discharge_spec"].groupby("time.dayofyear").mean()
        )
        climatology_doy = create_shape_aligned_climatology(
            benchmark_calculation_ds,
            climatology_doy.to_dataset(),
            variable="discharge_spec",
            time_period="dayofyear",
        )

        climatology_mon = (
            benchmark_calculation_ds["discharge_spec"].groupby("time.month").mean()
        )
        climatology_mon = create_shape_aligned_climatology(
            benchmark_calculation_ds,
            climatology_mon.to_dataset(),
            variable="discharge_spec",
            time_period="month",
        )

        all_preds["climatology_doy"] = climatology_doy.sel(
            station_id=all_preds.station_id, time=all_preds.time
        )["discharge_spec"]
        all_preds["climatology_mon"] = climatology_mon.sel(
            station_id=all_preds.station_id, time=all_preds.time
        )["discharge_spec"]

        return all_preds

    @staticmethod
    def calc_kratzert_error_functions(
        all_preds: xr.Dataset, metrics: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        all_metrics = [
            "nse",
            "mse",
            "rmse",
            "kge",
            "nse",
            "nse",
            "r",
            "fhv",
            "fms",
            "flv",
            "timing",
        ]
        assert all(
            np.isin(metrics, all_metrics)
        ), f"Metrics should be one of {all_metrics}"
        if metrics is None:
            metrics = all_metrics

        #  FOR USING THE KRATZERT FUNCTIONS (takes a long time)
        model_results = defaultdict(dict)
        for model in [v for v in all_preds.data_vars if v != "obs"]:
            for sid in tqdm(all_preds.station_id.values, desc=model):
                sim = all_preds[model].sel(station_id=sid).drop("station_id")
                obs = all_preds["obs"].sel(station_id=sid).drop("station_id")
                try:
                    model_results[model][sid] = calculate_metrics(
                        obs, sim, datetime_coord="time", metrics=metrics
                    )
                except ValueError:
                    model_results[model][sid] = np.nan

        results = {}
        for model in [k for k in model_results.keys()]:
            model_df = pd.DataFrame(model_results[model]).T
            results[model] = model_df

        return results

    def kratzert_errors(self, all_preds: xr.Dataset) -> Dict[str, pd.DataFrame]:
        assert all(np.isin(["LSTM", "EALSTM"], [v for v in all_preds.data_vars]))
        results = self.calc_kratzert_error_functions(all_preds)

        lstm_delta_dict = self.calculate_all_kratzert_deltas(results, ref_model="LSTM")
        lstm_delta = self.get_formatted_dataframe(lstm_delta_dict, format="metric")

        ealstm_delta_dict = self.calculate_all_kratzert_deltas(
            results, ref_model="EALSTM"
        )
        ealstm_delta = self.get_formatted_dataframe(ealstm_delta_dict, format="metric")
        return lstm_delta, ealstm_delta

    @staticmethod
    def calculate_all_kratzert_deltas(
        kratzert_results: Dict[str, pd.DataFrame], ref_model: str = "LSTM"
    ) -> DefaultDict[str, Dict[str, pd.Series]]:
        assert ref_model in [k for k in kratzert_results.keys()]
        assert model in [k for k in kratzert_results.keys()]
        ref_data = kratzert_results[ref_model]

        delta_dict = defaultdict(dict)
        # for each model calculate the difference for those metrics
        for model in [k for k in kratzert_results.keys() if k != ref_model]:
            model_data = kratzert_results[model]

            #  for each metric calculate either difference of absolute diffrerence (bias)
            for metric in ref_data.columns:
                ref = ref_data.loc[:, metric]
                m_data = model_data.loc[:, metric]
                if any(np.isin([metric], ["FHV", "FMS", "FLV"])):
                    result = ref.abs() - m_data.abs()
                else:
                    result = ref - m_data
                delta_dict[model][metric] = result

        return delta_dict

    def get_formatted_dataframe(
        delta_dict: DefaultDict[str, Dict[str, pd.Series]], format_: str = "metric"
    ) -> Dict[str, pd.DataFrame]:
        deltas = {}
        if format_ == "":
            for model in delta_dict.keys():
                deltas[model] = pd.DataFrame(delta_dict[model])
        elif format_ == "metric":
            metric_deltas = self.swap_nested_keys(delta_dict)
            for metric in metric_deltas.keys():
                deltas[metric] = pd.DataFrame(metric_deltas[metric])
        else:
            raise NotImplementedError

        return deltas

    @staticmethod
    def swap_nested_keys(original_dict) -> DefaultDict:
        #  https://stackoverflow.com/q/49333339/9940782
        #  move inner keys to outer keys and outer keys to inner
        new_dict = defaultdict(dict)
        for key1, value1 in original_dict.items():
            for key2, value2 in value1.items():
                new_dict[key2].update({key1: value2})
        return new_dict

    def _join_into_one_ds(self, ealstm_preds, lstm_preds, fuse_data) -> xr.Dataset:
        all_preds = xr.combine_by_coords(
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
        return all_preds

    @staticmethod
    def calculate_all_errors(
        all_preds: xr.DataArray,
        desc: str = None,
        metrics: List[str] = ["nse", "kge", "log_nse", "abs_pct_bias", "inv_kge", "mam30_ape"],
    ) -> Dict[str, pd.DataFrame]:
        station_names = pd.DataFrame(gauge_name_lookup, index=["gauge_name"]).T

        output_dict = defaultdict(list)
        station_names = pd.DataFrame(gauge_name_lookup, index=["gauge_name"]).T

        # Calculate Model Error Metrics for each model
        output_dict = defaultdict(list)
        station_names = pd.DataFrame(gauge_name_lookup, index=["gauge_name"]).T
        for ix, model in tqdm(
            enumerate([v for v in all_preds.data_vars if v != "obs"]), desc=desc
        ):
            _errors = calculate_errors(
                all_preds[["obs", model]].rename({model: "sim"})
            ).set_index("station_id")

            for metric in metrics:
                output_dict[metric].append(
                    _errors.rename({metric: model}, axis=1)[model]
                )

        # merge into single dataframe
        errors_dict = {}
        for metric in metrics:
            errors_dict[metric] = pd.concat(output_dict[metric], axis=1)

        return errors_dict

    @staticmethod
    def calculate_error_diff(
        error_df: pd.DataFrame, ref_model: str = "LSTM"
    ) -> pd.DataFrame:
        all_deltas = []
        for model in [c for c in error_df.columns if c != ref_model]:
            delta = error_df[ref_model] - error_df[model]
            delta.name = model
            all_deltas.append(delta)

        delta_df = pd.concat(all_deltas, axis=1)

        return delta_df

    def calculate_all_delta_dfs(
        self, errors_dict: Dict[str, pd.DataFrame]
    ) -> Tuple[Dict[str, pd.DataFrame]]:
        lstm_delta: Dict[str, pd.DataFrame] = defaultdict()
        ealstm_delta: Dict[str, pd.DataFrame] = defaultdict()

        for metric in [k for k in errors_dict.keys()]:
            if "bias" in metric:
                lstm_delta[metric] = self.calculate_error_diff(
                    errors_dict["bias"].abs(), ref_model="LSTM"
                )
                ealstm_delta[metric] = self.calculate_error_diff(
                    errors_dict["bias"].abs(), ref_model="EALSTM"
                )
            else:
                lstm_delta[metric] = self.calculate_error_diff(
                    error_df=errors_dict[metric], ref_model="LSTM"
                )
                ealstm_delta[metric] = self.calculate_error_diff(
                    error_df=errors_dict[metric], ref_model="EALSTM"
                )

        return lstm_delta, ealstm_delta

    @staticmethod
    def calculate_seasonal_deltas(
        self, all_preds: xr.Dataset,
    ) -> DefaultDict[str, Dict[str, Dict[str, pd.DataFrame]]]:
        seasonal_deltas = defaultdict(dict)
        for season in ["DJF", "MAM", "JJA", "SON"]:
            _preds = all_preds.sel(time=all_preds["time.season"] == season)
            seasonal_errors = self.calculate_all_errors(_preds, desc=season)
            (
                seasonal_deltas[season]["LSTM"],
                seasonal_deltas[season]["EALSTM"],
            ) = self.calculate_all_delta_dfs(seasonal_errors)
            seasonal_deltas[season]["raw"] = seasonal_errors

        return seasonal_deltas


def calculate_error_diff(
    error_df: pd.DataFrame, ref_model: str = "LSTM"
) -> pd.DataFrame:
    all_deltas = []
    for model in [c for c in error_df.columns if c != ref_model]:
        delta = error_df[ref_model] - error_df[model]
        delta.name = model
        all_deltas.append(delta)

    delta_df = pd.concat(all_deltas, axis=1)

    return delta_df


def calculate_all_delta_dfs(
    errors_dict: Dict[str, pd.DataFrame]
) -> Tuple[Dict[str, pd.DataFrame]]:

    lstm_delta: Dict[str, pd.DataFrame] = defaultdict()
    ealstm_delta: Dict[str, pd.DataFrame] = defaultdict()

    for metric in [k for k in errors_dict.keys()]:
        if metric == "bias":
            lstm_delta[metric] = calculate_error_diff(
                errors_dict["bias"].abs(), ref_model="LSTM"
            )
            ealstm_delta[metric] = calculate_error_diff(
                errors_dict["bias"].abs(), ref_model="EALSTM"
            )
        else:
            lstm_delta[metric] = calculate_error_diff(
                error_df=errors_dict[metric], ref_model="LSTM"
            )
            ealstm_delta[metric] = calculate_error_diff(
                error_df=errors_dict[metric], ref_model="EALSTM"
            )

    return lstm_delta, ealstm_delta


if __name__ == "__main__":
    save = True
    # LOAD IN DATA
    data_dir = Path("/cats/datastore/data")
    all_preds = xr.open_dataset(data_dir / "RUNOFF/all_preds.nc")

    # Calculate all errors
    all_errors = calculate_all_data_errors(all_preds, decompose_kge=True)
    all_metrics = get_metric_dataframes_from_output_dict(all_errors)

    metrics = ["pbias", "sqrt_kge", "sqrt_bias_ratio", "inv_variability_ratio", "variability_ratio", "correlation", "bias_ratio", "bias_error", "std_error"]
    calculated_metrics = [k for k in all_metrics.keys()]
    if not all(np.isin(metrics, calculated_metrics)):
        notin = np.array(metrics)[~np.isin(metrics, calculated_metrics)]
        assert False, print(f"{notin} not found in {calculated_metrics}")

    if save:
        import pickle
        pickle.dump(all_errors, (data_dir / "RUNOFF/all_errors.pkl").open("wb"))
        pickle.dump(all_metrics, (data_dir / "RUNOFF/all_metrics.pkl").open("wb"))

    # calculate delta errors
    calculate_all_delta_dfs(all_errors)

    # calculate seasonal delta errors
