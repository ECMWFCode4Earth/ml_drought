import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, Any, List


def plot_area_scatter(
    df: pd.DataFrame,
    region: str,
    metrics_df: Optional[pd.DataFrame] = None,
    ax=None,
    target_var: str = "VCI3M",
    region_name: Optional[str] = None,
    area_col_str: str = "region_name",
):
    # select station
    d = df.query(f"{area_col_str} == '{region}'").drop(columns=area_col_str)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = plt.gcf()

    # plot scatter
    ax.plot(d[target_var], d["preds"], "kx", alpha=0.6, label="Data Point")
    # plot 1:1 line
    line_1_1_x = np.linspace(d[target_var].min(), d[target_var].max(), 10)
    ax.plot(line_1_1_x, line_1_1_x, "k--", label="1:1 Line")

    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    title = (
        f"Station {region}" + f" {region_name}"
        if region_name is not None
        else f"Station {region}"
    )
    ax.set_title(title)

    ax.legend()

    if False:
        # making the plot pretty
        for item in (
            [ax.title, ax.xaxis.label, ax.yaxis.label]
            + ax.get_xticklabels()
            + ax.get_yticklabels()
        ):
            item.set_fontsize(12)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    return fig, ax


def plot_timeseries(
    df: pd.DataFrame,
    region: str,
    metrics_df: Optional[pd.DataFrame] = None,
    ax=None,
    region_name: Optional[str] = None,
    plot_years: Optional[List[int]] = None,
    area_col_str: str = "region_name",
):
    """Plot the Observed vs. Preds for the region"""
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()

    # plot the station
    if plot_years is None:
        df.query(f"{area_col_str} == '{region}'").drop(columns=area_col_str).plot(ax=ax)
    else:
        (
            df.loc[np.isin(df.index.year, plot_years)]
            .query(f"{area_col_str} == '{region}'")
            .drop(columns=area_col_str)
            .plot(ax=ax)
        )

    # get the error metrics
    if metrics_df is None:
        station_title = f"{region}" if region_name is not None else region
        ax.set_title(f"{station_title}")
    else:
        rmse_val = metrics_df.query(f"region == '{region}'").rmse.values[0]
        r2_val = metrics_df.query(f"region == '{region}'").r2.values[0]
        #  nse_val = metrics_df.query(f"region == '{region}'").nse.values[0]
        # set the title
        station_title = f"{region} {region_name}" if region_name is not None else region
        ax.set_title(
            f"{station_title}\nRMSE: {rmse_val:.2f} R2: {r2_val:.2f} NSE: {nse_val:.2f}"
        )

    return fig, ax


if __name__ == "__main__":
    # test_stations = ['22007', '27049', '28018', '31021', '31023', '34004', '35003', '39022', '41029', '51001', '55025', '57004', '83010']
    # regions = all_gdf.region_name.unique()
    regions = ["TURKANA", "MARSABIT", "WAJIR", "MANDERA"]
    model = "ealstm"
    plot_years = [2016]
    scale = 0.8
    fig, axs = plt.subplots(
        len(regions), 2, figsize=((12 * scale), (6 * scale) * len(regions))
    )

    for ix, region in enumerate(regions):
        try:
            plot_timeseries(
                df.query(f"model == '{model}'"),
                region,
                metrics_df=None,
                ax=axs[ix, 0],
                region_name=region,
                plot_years=plot_years,
            )
            plot_area_scatter(
                df.query(f"model == '{model}'"), region, metrics_df=None, ax=axs[ix, 1]
            )
        except TypeError:
            print(f"** {region_name} data does not exist in the predictions! **")

        plt.tight_layout()
