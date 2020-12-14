import pandas as pd
from typing import Tuple, List, Dict, Optional
from collections import defaultdict
from scripts.utils import get_data_path
from pathlib import Path
import sys

sys.path.append("/home/tommy/ml_drought")
from scripts.drafts.calculate_error_scores import (
    calculate_all_data_errors,
    get_metric_dataframes_from_output_dict,
)
from scripts.drafts.calculate_error_scores import DeltaError


def skill_score(model, bench, perfect):
    return (model - bench) / (perfect - bench)


def create_skill_score(
    all_metrics: pd.DataFrame,
    metric: str,
    benchmark: str = "climatology_doy",
    models: List[str] = ["TOPMODEL", "ARNOVIC", "PRMS", "SACRAMENTO", "EALSTM", "LSTM"],
) -> pd.DataFrame:
    perf_lookup: Dict[str, float] = dict(
        kge=1, nse=1, inv_kge=1, log_nse=1, mse=0, mape=0, mam30_ape=0, rmse=0
    )
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


if __name__ == "__main__":
    data_dir = Path("/cats/datastore/data/")

    #  calculate all metrics
    processor = DeltaError(ealstm_preds, lstm_preds, fuse_data)
    all_preds = processor.all_preds
    all_errors = calculate_all_data_errors(all_preds)
    all_metrics = get_metric_dataframes_from_output_dict(all_errors)

    #  calculate skill scores
    kge_ss = create_skill_score(all_metrics, metric="kge", benchmark="climatology_doy")
    inv_kge_ss = create_skill_score(
        all_metrics, metric="inv_kge", benchmark="climatology_doy"
    )
