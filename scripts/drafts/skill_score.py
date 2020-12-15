from typing import Tuple, List, Dict, Optional, DefaultDict
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
import sys

sys.path.append("/home/tommy/ml_drought")
from scripts.utils import get_data_path
from scripts.drafts.calculate_error_scores import (
    calculate_all_data_errors,
    get_metric_dataframes_from_output_dict,
)
from scripts.drafts.calculate_error_scores import DeltaError


perf_lookup: Dict[str, float] = dict(
    kge=1, nse=1, inv_kge=1, log_nse=1, mse=0, mape=0, mam30_ape=0, rmse=0
)


def skill_score(model, bench, perfect):
    return (model - bench) / (perfect - bench)


def create_skill_score(
    all_metrics: pd.DataFrame,
    metric: str,
    benchmark: str = "climatology_doy",
    models: List[str] = ["TOPMODEL", "ARNOVIC", "PRMS", "SACRAMENTO", "EALSTM", "LSTM"],
) -> pd.DataFrame:
    assert (
        metric in perf_lookup.keys()
    ), f"Expected {metric} to be in {perf_lookup.keys()}"

    metric_ss = defaultdict(list)
    metric_df = all_metrics[metric]
    bench = metric_df[benchmark]
    for model in models:
        metric_ss[model] = skill_score(metric_df[model], bench, perf_lookup[metric])

    metric_ss = pd.DataFrame(metric_ss)

    return metric_ss


def create_all_skill_scores(
    all_metrics: pd.DataFrame,
    benchmark: str = "climatology_doy",
    metrics: List[str] = ["kge", "inv_kge", "nse", "log_nse"],
    models: List[str] = ["TOPMODEL", "ARNOVIC", "PRMS", "SACRAMENTO", "EALSTM", "LSTM"],
) -> DefaultDict[str, pd.DataFrame]:
    skill_score_dict = defaultdict(dict)
    assert all(
        np.isin(metrics, [l for l in all_metrics.keys()])
    ), f"Expect {metrics} to be in {all_metrics.keys()}"

    for metric in metrics:
        skill_score_dict[metric] = create_skill_score(
            all_metrics, metric=metric, benchmark=benchmark, models=models
        )
    return skill_score_dict


if __name__ == "__main__":
    import xarray as xr

    data_dir = Path("/cats/datastore/data/")

    #  calculate all metrics
    all_preds = xr.open_dataset(data_dir / "RUNOFF/all_preds.nc")
    all_errors = calculate_all_data_errors(all_preds)
    all_metrics = get_metric_dataframes_from_output_dict(all_errors)

    #  calculate skill scores
    skill_score_dict = create_all_skill_scores(all_metrics, benchmark="climatology_doy")

    kge_ss = create_skill_score(all_metrics, metric="kge", benchmark="climatology_doy")
    inv_kge_ss = create_skill_score(
        all_metrics, metric="inv_kge", benchmark="climatology_doy"
    )
