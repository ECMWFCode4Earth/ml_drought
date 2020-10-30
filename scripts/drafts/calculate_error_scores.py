import pandas as pd
import numpy as np
import xarray as xr

from src.analysis.evaluation import spatial_rmse, spatial_r2, spatial_nse, spatial_bias, spatial_kge
from src.analysis.evaluation import temporal_rmse, temporal_r2, temporal_nse
from src.analysis.evaluation import _nse_func, _rmse_func, _r2_func, _bias_func, _kge_func, _mse_func
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
    df = df.dropna(how='any')
    df = df.reset_index().set_index("time")

    station_ids = df["station_id"].unique()
    errors = []
    for station_id in station_ids:
        d = df.loc[df["station_id"] == station_id]
        if error_str == "rmse":
            _error_calc = error_func(
                d["obs"].values, d["sim"].values, n_instances=d.size)
        else:
            _error_calc = error_func(d["obs"].values, d["sim"].values)
        errors.append(_error_calc)

    error = pd.DataFrame({"station_id": station_ids, error_str: errors})

    return error


def calculate_ml_errors(preds: xr.Dataset) -> pd.DataFrame:
    errors = [
        error_func(preds, "nse").set_index('station_id'),
        error_func(preds, "kge").set_index('station_id'),
        error_func(preds, "mse").set_index('station_id'),
        error_func(preds, "bias").set_index('station_id'),
    ]
    error_df = errors[0].join(errors[1].join(
        errors[2]).join(errors[3])).reset_index()

    return error_df

class FuseErrors:
    def __init__(self, fuse_data: xr.Dataset):
        assert all(np.isin(
            ["obs", "SimQ_TOPMODEL", "SimQ_PRMS", "SimQ_ARNOVIC",
                "SimQ_SACRAMENTO"], [v for v in fuse_data.data_vars]
        ))

        self.fuse_data = fuse_data
        self._separate_into_das()

        nse_df = self._calculate_metric("nse")
        kge_df = self._calculate_metric("kge")
        bias_df = self._calculate_metric("bias")
        rmse_df = self._calculate_metric("rmse")

        # convert into one clean dataframe
        fuse_errors = pd.concat([nse_df, rmse_df, r2_df, bias_df], axis=1)
        try:
            fuse_errors = fuse_errors.drop(
                'time', axis=1, level=1).swaplevel(axis=1).sort_index(axis=1)
        except KeyError:
            pass

        self.fuse_errors = fuse_errors.rename({"NSE": "nse", "BIAS": "bias", "MSE": "mse"}, axis=1, level=0)


    def _separate_into_das(self) -> None:
        # separate into DataArrays
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
            "kge": spatial_kge
        }
        function = metric_lookup[metric]

        out_list = []
        for model, model_name in zip(self.model_preds, self.model_names):
            out_list.append(function(self.obs, model).rename(model_name))

        metric_xr = xr.merge([
            out_list[0],
            out_list[1],
            out_list[2],
            out_list[3],
        ])
        metric_df = metric_xr.to_dataframe()
        metric_df = static['gauge_name'].to_dataframe().join(metric_df).rename(columns=dict(gauge_name="Name"))
        metric_df.columns = [[metric for _ in range(len(metric_df.columns))], metric_df.columns]

        return metric_df

