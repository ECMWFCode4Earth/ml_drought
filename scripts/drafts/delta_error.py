import pandas as pd
import numpy as np
import xarray as xr
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from functools import partial
from typing import Dict, DefaultDict, Tuple, List, Optional

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
            "alpha-nse",
            "beta-nse",
            "fhv",
            "fms",
            "flv",
            "peak-timing",
            "pearson-r",
        ]
        if metrics is None:
            metrics = all_metrics

        assert all(
            np.isin(metrics, all_metrics)
        ), f"Metrics should be one of {all_metrics}"

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
        metrics: List[str] = [
            "nse",
            "kge",
            "log_nse",
            "abs_pct_bias",
            "inv_kge",
            "mam30_ape",
        ],
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
        self, errors_dict: Dict[str, pd.DataFrame], absolute_metrics: List[str] = [],
    ) -> Tuple[Dict[str, pd.DataFrame]]:
        lstm_delta: Dict[str, pd.DataFrame] = defaultdict()
        ealstm_delta: Dict[str, pd.DataFrame] = defaultdict()

        for metric in [k for k in errors_dict.keys()]:
            if metric in absolute_metrics:
                lstm_delta[metric] = self.calculate_error_diff(
                    errors_dict[metric].abs(), ref_model="LSTM"
                )
                ealstm_delta[metric] = self.calculate_error_diff(
                    errors_dict[metric].abs(), ref_model="EALSTM"
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


if __name__ == "__main__":
    from scripts.drafts.io_results import read_ensemble_results, read_fuse_data
    from scripts.drafts.calculate_error_scores import (
        get_metric_dataframes_from_output_dict,
    )

    save = True
    data_dir = Path("/cats/datastore/data")

    pet_ealstm_ensemble_dir = data_dir / "runs/ensemble_pet_ealstm"
    ealstm_preds = read_ensemble_results(pet_ealstm_ensemble_dir)

    # lstm_ensemble_dir = data_dir / "runs/ensemble"
    lstm_ensemble_dir = data_dir / "runs/ensemble_pet"
    lstm_preds = read_ensemble_results(lstm_ensemble_dir)

    #  fuse data
    raw_fuse_path = data_dir / "RUNOFF/FUSE"
    fuse_data = read_fuse_data(raw_fuse_path, lstm_preds["obs"])

    processor = DeltaError(ealstm_preds, lstm_preds, fuse_data, incl_benchmarks=False)
    kratzert_models = processor.calc_kratzert_error_functions(
        processor.all_preds, metrics=None
    )
    kratzert_metrics = get_metric_dataframes_from_output_dict(kratzert_models)

    if save:
        import pickle

        pickle.dump(
            kratzert_models, (data_dir / "RUNOFF/kratzert_models.pkl").open("wb")
        )
        pickle.dump(
            kratzert_metrics, (data_dir / "RUNOFF/kratzert_metrics.pkl").open("wb")
        )
