from typing import Optional, List, Dict, Tuple
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys

sys.path.append("/home/tommy/ml_drought")
from scripts.drafts.ml_sids import ml_sids

#  --------------------------------- #
#  CDF Plot #
#  --------------------------------- #
def plot_cdf(
    error_data,
    metric: str = "",
    sids: List[int] = ml_sids,
    clip: Optional[Tuple] = None,
    ax=None,
    title=None,
    models: Optional[List[str]] = None,
    median: bool = True,
):
    colors = sns.color_palette()
    kwargs_dict = {
        "TOPMODEL": {"linewidth": 1, "alpha": 0.8, "color": colors[2], "clip": clip},
        "PRMS": {"linewidth": 1, "alpha": 0.8, "color": colors[3], "clip": clip},
        "ARNOVIC": {"linewidth": 1, "alpha": 0.8, "color": colors[4], "clip": clip},
        "VIC": {"linewidth": 1, "alpha": 0.8, "color": colors[4], "clip": clip},
        "SACRAMENTO": {"linewidth": 1, "alpha": 0.8, "color": colors[5], "clip": clip},
        "gr4j": {"linewidth": 1, "alpha": 0.8, "color": colors[9], "clip": clip},
        "climatology": {
            "linewidth": 1,
            "alpha": 0.8,
            "color": colors[6],
            "clip": clip,
            "ls": "-.",
        },
        "climatology_doy": {
            "linewidth": 1,
            "alpha": 0.8,
            "color": colors[6],
            "clip": clip,
            "ls": "-.",
        },
        "climatology_mon": {
            "linewidth": 1,
            "alpha": 0.8,
            "color": colors[8],
            "clip": clip,
            "ls": "-.",
        },
        "persistence": {
            "linewidth": 1,
            "alpha": 0.8,
            "color": colors[7],
            "clip": clip,
            "ls": "-.",
        },
        "EALSTM": {"linewidth": 3, "alpha": 1, "color": colors[1], "clip": clip},
        "LSTM": {"linewidth": 3, "alpha": 1, "color": colors[0], "clip": clip},
    }
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 3))

    if models is None:
        models = [c for c in error_data.columns if c in kwargs_dict]
    for ix, model in enumerate(models):
        summary_stat = (
            error_data[model].dropna().median()
            if median
            else error_data[model].dropna().mean()
        )
        sns.kdeplot(
            error_data[model].dropna(),
            cumulative=True,
            legend=False,
            ax=ax,
            label=f"{model}: {summary_stat:.2f}",
            **kwargs_dict[model],
        )

        ax.axvline(summary_stat, ls="--", color=kwargs_dict[model]["color"])

    ax.set_xlim(clip)
    ax.set_xlabel(metric)
    ax.set_ylabel("Cumulative density")
    title = (
        title
        if title is not None
        else f"Cumuluative Density Function of Station {metric} Scores"
    )
    ax.set_title(title)
    sns.despine()
    plt.legend()

    return ax


#  --------------------------------- #
# Budyko Curve #
#  --------------------------------- #


def assign_wateryear(dt):
    """https://stackoverflow.com/a/52615358/9940782"""
    dt = pd.Timestamp(dt)
    if dt.month >= 10:
        return pd.datetime(dt.year + 1, 1, 1).year
    else:
        return pd.datetime(dt.year, 1, 1).year


def curve(x):
    return 1 - (1 / x)


def wetness_ix(p, pe):
    return p / pe


def runoff_coeff(p, q):
    return q / p


def calculate_curve_params(ds: xr.Dataset):
    x = wetness_ix(ds["precipitation"], ds["pet"])
    y = runoff_coeff(ds["precipitation"], ds["discharge_spec"])

    return x, y


def _add_annotations(ax):
    ax.text(
        8,
        0.1,
        "Runoff Deficits\nExceed Total PET",
        size=10,
        rotation=0,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round", ec=(1.0, 0.5, 0.5), fc=(1.0, 0.8, 0.8),),
    )

    ax.text(
        5,
        1.4,
        "Runoff Exceeds Rainfall Inputs",
        size=10,
        rotation=0,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round", ec=(1.0, 0.5, 0.5), fc=(1.0, 0.8, 0.8),),
    )
    return ax


def plot_budyko_curve(
    ds: xr.Dataset,
    color_var: Optional[np.ndarray] = None,
    color_label: str = "",
    ax=None,
    scale: float = 1.2,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    annotate: bool = True,
    colorbar: bool = True,
    scatter_kwargs: Dict = {"alpha": 0.7},
):
    #  1. calculate wetness index (x) and runoff coefficient (y)
    assert all(np.isin(["precipitation", "pet", "discharge_spec"], ds.data_vars))
    x, y = calculate_curve_params(ds=ds)

    if ax == None:
        _, ax = plt.subplots(figsize=(6 * scale, 4 * scale))

    # set colors
    if color_var is not None:
        if vmin is None:
            vmin = color_var.quantile(q=0.15)
        if vmax is None:
            vmax = color_var.quantile(q=0.85)

    # 2. create the scatter plot
    sc = ax.scatter(x, y, c=color_var, vmin=vmin, vmax=vmax, **scatter_kwargs)
    if (color_var is not None) and colorbar:
        cbar = plt.colorbar(sc)
        cbar.ax.set_ylabel(color_label)

    #  3. create the reference lines
    #  horizontal line
    ax.axhline(1, color="grey", ls="--")
    # curve
    c = curve(np.linspace(1e-1, ax.get_xlim()[-1] + 1))
    ax.plot(np.linspace(1e-1, ax.get_xlim()[-1] + 1), c, color="grey", ls="-.")

    #  4. beautify the plot
    if annotate:
        ax = _add_annotations(ax)

    ax.set_ylim(0, 1.5)
    ax.set_xlim(0, 9)
    ax.set_xlabel("Wetness Index (P/PE)")
    ax.set_ylabel("Runoff Coefficient (Q/P)")
    sns.despine()

    return ax


if __name__ == "__main__":
    from pathlib import Path
    from scripts.drafts.calculate_error_scores import (
        DeltaError,
        calculate_all_data_errors,
        get_metric_dataframes_from_output_dict,
    )

    # LOAD IN DATA
    data_dir = Path("/cats/datastore/data")
    ds = xr.open_dataset(data_dir / "RUNOFF/ALL_dynamic_ds.nc")
    ds["station_id"] = ds["station_id"].astype(int)

    # predictions
    all_preds = xr.open_dataset(data_dir / "RUNOFF/all_preds.nc")
    all_errors = calculate_all_data_errors(all_preds)
    all_metrics = get_metric_dataframes_from_output_dict(all_errors)

    #  SET PLT OPTIONS
    label_size = 14  #  10
    plt.rcParams.update(
        {"axes.labelsize": label_size, "legend.fontsize": label_size, "font.size": 14,}
    )

    # 1. Plot NSE CDF
    f, ax = plt.subplots(figsize=(12, 8))
    plot_cdf(
        all_metrics["nse"], metric="NSE", title="", ax=ax, clip=(0, 1), median=True
    )
    f.savefig(data_dir / "RUNOFF/cdf_all.png")

    #  2. Plot the Budyko Curve
    _ds = ds.sel(station_id=all_metrics["nse"]["LSTM"].index).mean(dim="time")
    color_var = all_metrics["mam30_ape"]["LSTM"]
    ax = None
    plot_budyko_curve(
        ds=_ds,
        color_var=color_var,
        color_label="LSTM mam30_ape",
        vmin=20,
        vmax=60,
        scatter_kwargs={"cmap": "viridis_r", "alpha": 0.7},
        ax=ax,
        annotate=False,
    )
    plt.gcf().savefig(data_dir / "RUNOFF/budyko_curve.png")