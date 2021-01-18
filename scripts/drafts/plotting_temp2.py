from typing import Tuple, Dict, Optional, List, Union, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_station_scatter(
    df: pd.DataFrame,
    station_id: str,
    metrics_df: Optional[pd.DataFrame] = None,
    ax=None,
    target_var: str = "discharge_spec",
    station_name: Optional[str] = None,
    color_by_season: bool = None,
):
    # select station
    d = df.query(f"station_id == '{station_id}'").drop(columns="station_id")

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = plt.gcf()

    if not color_by_season:
        # plot scatter
        ax.plot(d[target_var], d.preds, "kx", alpha=0.6, label="Data Point")
    else:
        seasons = ds.sel(time=d.index)["time.season"].values
        d["season"] = seasons
        for ix, season in enumerate(d.season.unique()):
            ax.scatter(
                d.loc[d["season"] == season, target_var],
                d.loc[d["season"] == season, "preds"],
                color=sns.color_palette()[ix],
                alpha=0.6,
                label=season,
                marker="x",
            )
            sns.regplot(
                d.loc[d["season"] == season, target_var],
                d.loc[d["season"] == season, "preds"],
                color=sns.color_palette()[ix],
                ax=ax,
                scatter=False,
                ci=None,
            )
    # plot 1:1 line
    max_val = max(ax.get_xlim()[-1], ax.get_ylim()[-1])
    line_1_1_x = np.linspace(0, max_val, 10)
    ax.plot(line_1_1_x, line_1_1_x, "k--", label="1:1 Line")

    # set the xylim
    ax.set_ylim(0, max_val)
    ax.set_xlim(0, max_val)

    ax.set_xlabel("Observed $[mm d^{-1} km^{-2}]$")
    ax.set_ylabel("Predicted $[mm d^{-1} km^{-2}]$")
    title = (
        f"Station {station_id}" + f" {station_name}"
        if station_name is not None
        else f"Station {station_id}"
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


def plot_station(
    df: pd.DataFrame,
    station_id: str,
    metrics_df: Optional[pd.DataFrame] = None,
    ax=None,
    station_name: Optional[str] = None,
    plot_years: Optional[List[int]] = None,
):
    """Plot the Observed vs. Preds for the station_id"""
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()

    # plot the station
    if plot_years is None:
        df.query(f"station_id == '{station_id}'").drop(columns="station_id").plot(ax=ax)
    else:
        (
            df.loc[np.isin(df.index.year, plot_years)]
            .query(f"station_id == '{station_id}'")
            .drop(columns="station_id")
            .plot(ax=ax)
        )

    # get the error metrics
    try:
        rmse_val = metrics_df.query(f"station_id == '{station_id}'").rmse.values[0]
    except AttributeError:
        rmse_val = np.nan
    nse_val = metrics_df.query(f"station_id == '{station_id}'").nse.values[0]
    # set the title
    station_title = (
        f"{station_id} {station_name}" if station_name is not None else station_id
    )
    ax.set_title(f"{station_title}\nRMSE: {rmse_val:.2f} NSE: {nse_val:.2f}")

    return fig, ax


def plot_catchment_time_series(
    df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    catchment_ids: List[str],
    catchment_names: List[str],
    plot_years: List[int] = [2011],
    scale: float = 0.8,
    color_by_season: bool = False,
):
    n_plots = len(catchment_ids)
    station_map = dict(zip(catchment_ids, catchment_names))
    fig, axs = plt.subplots(n_plots, 2, figsize=(12 * scale, 6 * scale * n_plots))

    for ix, (station_id, station_name) in enumerate(
        zip(catchment_ids, catchment_names)
    ):
        #     fig, axs = plt.subplots(1, 2, figsize=(12*scale, 6*scale))
        try:
            plot_station(
                df,
                station_id,
                metrics_df,
                ax=axs[ix, 0],
                station_name=station_name,
                plot_years=plot_years,
            )
            plot_station_scatter(
                df, station_id, metrics_df, axs[ix, 1], color_by_season=color_by_season
            )
        except IndexError:
            #  the axes are one dimensional
            plot_station(
                df,
                station_id,
                metrics_df,
                ax=axs[0],
                station_name=station_name,
                plot_years=plot_years,
            )
            plot_station_scatter(
                df, station_id, metrics_df, axs[1], color_by_season=color_by_season
            )
        except TypeError:
            print(f"** {station_name} data does not exist in the predictions! **")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig, axs
