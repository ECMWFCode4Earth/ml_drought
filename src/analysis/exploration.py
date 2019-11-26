"""
Some handy data exploration functions
"""
import xarray as xr
import pandas as pd
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt

from typing import Optional, Tuple
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def calculate_seasonal_anomalies(ds: xr.Dataset, variable: str) -> xr.DataArray:
    """create a DataArray of the SEASONAL anomalies (difference from seasonal
    mean).

    Arguments:
    ---------
    ds: xr.Dataset
        the input dataset

    variable: str
        the variable that you want to calculate anomalies from
    """
    # calculate seasonal values
    resample_da = ds[variable].resample(time="Q-DEC").mean()
    resample_da["season"] = resample_da["time.season"]
    # calculate climatology (of same shape)
    climatology = resample_da.groupby("time.season").mean()
    climatology_ = xr.ones_like(resample_da)
    climatology_.values = [
        climatology.sel(season=season).values for season in resample_da.season
    ]
    climatology_.name = "climatology"
    # join to da
    ds = xr.merge([resample_da, climatology_])
    anomaly = ds[variable] - ds["climatology"]
    return anomaly.rename("anomaly")


def calculate_seasonal_anomalies_spatial(ds: xr.Dataset, variable: str) -> xr.DataArray:
    """create a DataArray of the SEASONAL anomalies (difference from seasonal
    mean).

    Arguments:
    ---------
    ds: xr.Dataset
        the input dataset

    variable: str
        the variable that you want to calculate anomalies from
    """
    # calculate seasonal values
    resample_da = ds[variable].resample(time="Q-DEC").mean(dim="time")
    resample_da["season"] = resample_da["time.season"]
    # calculate climatology (of same shape)
    climatology = resample_da.groupby("time.season").mean(dim="time")
    climatology_ = xr.ones_like(resample_da)
    climatology_.values = [
        climatology.sel(season=season).values for season in resample_da.season.values
    ]
    climatology_.name = "climatology"
    # join to da
    ds = xr.merge([resample_da, climatology_])
    anomaly = ds[variable] - ds["climatology"]
    return anomaly.rename("anomaly")


def create_anomaly_df(
    anomaly_da: xr.DataArray,
    mintime: Optional[str] = None,
    maxtime: Optional[str] = None,
) -> pd.DataFrame:
    """From the anomaly xr.DataArray created in `calculate_seasonal_anomalies()`
    create an anomaly dataframe with `time` and `anomaly` columns. This is
    easier for working with the plotting functions in `src.analysis.evaluation`.

    Arguments:
    ---------
    anomaly_da: xr.DataArray
        the anomaly dataArray from `calculate_seasonal_anomalies()`

    mintime: Optional[str] = None
        if not None then selects a minimum time (for selecting a date range)

    maxtime: Optional[str] = None
        if not None then selects a maximum time (for selecting a date range)
    """
    df = (
        anomaly_da.sel(time=slice(mintime, maxtime))  # type: ignore
        .to_pandas()
        .to_frame("anomaly")
        .reset_index()
        .astype({"time": "datetime64[ns]"})
    )
    return df


def plot_bar_anomalies(
    df: pd.DataFrame,
    variable: str = "anomaly",
    ax: Optional[Axes] = None,
    positive_color: str = "b",
    negative_color: str = "r",
) -> Tuple[Axes, Figure]:
    """Plot the seasonal anomalies calculated in
    `create_anomaly_df()` with positive/negative
    anomalies as bars. Also with sensible xlabels

    Arguments:
    ---------
    df: pd.DataFrame

    variable: str = 'anomaly'

    ax: Optional[Axes] = None
    """
    # https://stackoverflow.com/a/30135182/9940782 - format date xticks
    # https://stackoverflow.com/a/22311398/9940782 - color the bars
    assert "time" in df.columns
    assert type(df["time"][0]) == pd._libs.tslibs.timestamps.Timestamp

    # create color
    df["positive"] = df[variable] > 0
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = plt.gcf()

    df.plot.bar(
        x="time",
        y=variable,
        ax=ax,
        color=df.positive.map({True: positive_color, False: negative_color}),
    )

    # fix the xaxis datelabels
    len_years = len(df.time.dt.year.unique())
    ticklabels = [""] * len(df)
    skip = len(df) // len_years
    ticklabels[::skip] = df["time"].iloc[::skip].dt.strftime("%Y")
    ax.xaxis.set_major_formatter(mticker.FixedFormatter(ticklabels))
    fig.autofmt_xdate()

    return fig, ax
