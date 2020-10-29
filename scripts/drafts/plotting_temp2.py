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
    line_1_1_x = np.linspace(d[target_var].min(), d[target_var].max(), 10)
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
