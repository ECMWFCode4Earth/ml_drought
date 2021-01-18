from typing import Tuple, Dict, Optional, List, Union, Any


def remove_invalid_vals(x, y) -> Tuple[np.ndarray, np.ndarray]:
    """check for infinite or nan values

    Required for np.polyfit:
    https://stackoverflow.com/a/13693657/9940782
    """
    isfinite = np.isfinite(y) & np.isfinite(x)
    notnull = pd.notnull(y) & pd.notnull(x)

    x = x[isfinite & notnull]
    y = y[isfinite & notnull]

    return x, y


def plot_1_1_line(x: np.ndarray, ax) -> plt.Axes:
    # plot 1:1 line
    line_1_1_x = np.linspace(x.min(), x.max(), 10)
    ax.plot(line_1_1_x, line_1_1_x, "k--", label="1:1 Line", alpha=0.5)
    return ax


def plot_scatter(
    x: np.ndarray, y: np.ndarray, ax, one_to_one: bool = True, **kwargs
) -> plt.Axes:
    """Scatter plot of x vs. y"""
    # plot scatter
    ax.plot(x, y, "kx", **kwargs)

    if one_to_one:
        # plot 1:1 line
        ax = plot_1_1_line(x, ax)

    return ax


def plot_reg_line(x: np.ndarray, y: np.ndarray, ax, auto_label: bool = True, **kwargs):
    """plot linear regression line of x vs. y"""
    # plot regression line
    x, y = remove_invalid_vals(x, y)
    m, b = np.polyfit(x, y, 1)
    reg = m * x + b
    if auto_label:
        label = f"Regression Line: {m:.2f}X + {b:.2f}"
        ax.plot(x, reg, label=label, **kwargs)
    else:
        ax.plot(x, reg, **kwargs)

    return ax


def plot_station_scatter(
    df: pd.DataFrame,
    station_id: str,
    metrics_df: Optional[pd.DataFrame] = None,
    ax=None,
    target_var: str = "discharge_spec",
    station_name: Optional[str] = None,
    color_by_season: bool = None,
):
    # select station & data
    d = df.query(f"station_id == '{station_id}'").drop(columns="station_id")

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = plt.gcf()

    if not color_by_season:
        x = d[target_var]
        y = d.preds

        # plot scatter
        kwargs = dict(alpha=0.6, label="Data Point")
        ax = plot_scatter(x, y, ax, **kwargs)

        # plot regression line
        kwargs = dict(color="#7bd250", ls="--")
        ax = plot_reg_line(x, y, ax=ax, **kwargs)
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
    line_1_1_x = np.linspace(x.min(), x.max(), 10)
    ax.plot(line_1_1_x, line_1_1_x, "k--", label="1:1 Line")

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

    if "month" in df.columns:
        df = df.drop(columns="month")

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
    rmse_val = metrics_df.query(f"station_id == '{station_id}'").rmse.values[0]
    r2_val = metrics_df.query(f"station_id == '{station_id}'").r2.values[0]
    nse_val = metrics_df.query(f"station_id == '{station_id}'").nse.values[0]
    # set the title
    station_title = (
        f"{station_id} {station_name}" if station_name is not None else station_id
    )
    ax.set_title(
        f"{station_title}\nRMSE: {rmse_val:.2f} R2: {r2_val:.2f} NSE: {nse_val:.2f}"
    )

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
