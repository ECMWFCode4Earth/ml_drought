"""
Scripts for plotting region objects
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import List, Optional, Tuple, Union

from geopandas.geodataframe import GeoDataFrame


def _plot_single_gdf(
    ax: Axes,
    gdf: GeoDataFrame,
    column_to_plot: str,
    title: Optional[str] = None,
    cmap: Optional[str] = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Axes:
    # nicely format the colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    gdf.plot(
        column_to_plot, ax=ax, legend=True, cmap=cmap, vmin=vmin, vmax=vmax, cax=cax
    )
    ax.set_title(title)
    return ax


def get_vrange(
    array1: Union[pd.Series, np.array],
    array2: Union[pd.Series, np.array],
    how: str = "percentile",
) -> Tuple[float, float]:
    if how == "percentile":
        vmin = min(np.nanpercentile(array1, 5), np.nanpercentile(array2, 5))
        vmax = max(np.nanpercentile(array1, 95), np.nanpercentile(array2, 95))
    else:
        print("calculating vmin/vmax using `simple` method")
        vmin = min(min(array1), min(array2))
        vmax = max(max(array1), max(array2))
    return vmin, vmax


def get_valid_metrics(gdf: GeoDataFrame) -> List[str]:
    valid_cols = [
        c.split("_")[-1] for c in gdf.columns if ("true_" in c) or ("pred_" in c)
    ]
    return np.unique(valid_cols)


def plot_comparison_maps(
    gdf: GeoDataFrame,
    metric: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    suptitle: Optional[str] = None,
) -> Tuple[Figure, List[Axes]]:
    """Plot the true / pred columns in a GeoDataFrame

    Arguments:
    ---------
    gdf: GeoDataFrame,
        GeoDataFrame storing the data for true/predicted data

    metric: str,
        the suffix for the true_ / pred_ columns to compare

    vmin: Optional[float] = None,
        whether to hardcode or automatically choose colorscale

    vmax: Optional[float] = None,
        whether to hardcode or automatically choose colorscale

    suptitle: Optional[str] = None
        title of the whole plot

    Expected gdf object:
    -------------------
    >>> gdf.columns
    Out[83]:
    Index(['datetime', 'region_name', 'pred_value', 'geometry',
    'true_value','pred_VDI', 'true_VDI'])

    >>> gdf.head().drop(columns='geometry')
    Out[90]:
        datetime region_name  pred_value  true_value  pred_VDI  true_VDI
    0 2018-02-28     NAIROBI   33.995075   36.565174         3         4
    1 2018-03-31     NAIROBI   41.027580   43.697745         4         4
    2 2018-04-30     NAIROBI   48.354313   49.558459         4         4
    """
    # check the metric is a valid one
    valid_metrics = get_valid_metrics(gdf)
    assert metric in valid_metrics, f"Try one of: {valid_metrics}"

    # build the colname for that metric
    true_data_colname = f"true_{metric}"
    pred_data_colname = f"pred_{metric}"

    # consistent colorbar
    if vmin is None:
        vmin, _ = get_vrange(
            gdf[true_data_colname], gdf[pred_data_colname], how="percentile"
        )
    if vmax is None:
        _, vmax = get_vrange(
            gdf[true_data_colname], gdf[pred_data_colname], how="percentile"
        )

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # true and predicted column names
    for i, colname in enumerate([true_data_colname, pred_data_colname]):
        assert colname in gdf.columns, f"Expecting to find {colname} in {gdf.columns}"

        ax = axs[i]
        ax = _plot_single_gdf(
            gdf=gdf, column_to_plot=colname, ax=ax, title=colname, vmin=vmin, vmax=vmax
        )

    if suptitle is not None:
        fig.suptitle(suptitle)
    return fig, axs
