from pathlib import Path
from typing import Dict, Callable, List
from collections import defaultdict
import pandas as pd

from scipy.stats import wilcoxon, ks_2samp


def _result_df(
    func: Callable, metric_df: pd.DataFrame, models: List[str], ref_model: str
) -> pd.DataFrame:
    results = defaultdict(dict)
    for model in models:
        res_ = func(metric_df[model], metric_df[ref_model])
        results[model]["statistic"] = res_.statistic
        results[model]["pvalue"] = res_.pvalue

    return pd.DataFrame(results)


def run_test(
    all_metrics: Dict[str, pd.DataFrame],
    test: str = "ks",
    metric: str = "nse",
    ref_model: str = "LSTM",
):
    assert test in ["ks", "wilcoxon"]
    lookup = {"ks": ks_2samp, "wilcoxon": wilcoxon}
    func = lookup[test]

    # build the dataframe of metrics (FUSE + LSTM + EALSTM)
    metric_df = all_metrics[metric]

    models = ["TOPMODEL", "PRMS", "SACRAMENTO", "ARNOVIC", "LSTM", "EALSTM"]
    ms = [m for m in models if m != ref_model]  #  run the test
    df = _result_df(func, metric_df, models=ms, ref_model=ref_model)
    # quick significance column
    df.loc["p0.01"] = (df.loc["pvalue"] < 1 * 1e-2).astype(bool)
    df.loc["p0.001"] = (df.loc["pvalue"] < 1 * 1e-3).astype(bool)
    return df


if __name__ == "__main__":
    import pickle

    data_dir = Path("/cats/datastore/data/")
    all_metrics = pickle.load((data_dir / "RUNOFF/all_metrics.pkl").open("rb"))
    seasonal_metrics = pickle.load(
        (data_dir / "RUNOFF/seasonal_metrics.pkl").open("rb")
    )

    print(run_test(all_metrics, test="ks", metric="nse", ref_model="LSTM"))
    print(run_test(all_metrics, test="wilcoxon", metric="nse", ref_model="LSTM"))

    print(run_test(seasonal_metrics["JJA"], test="ks", metric="nse"))
