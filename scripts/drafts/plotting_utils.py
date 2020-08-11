"""
Rough plotting routines

* Histograms (Marginal Distributions)
* Joint Distributions
* Temporal Plots / timeseries
* Spatio-temporal plots
* Spatial Plots

NOTE: NEED TO STANDARDISE THIS API!
-  are inputs data array or dataset?
-  do you provide a fig,ax or does it create one for you?
-  REMOVE hardcoding of figure labels
"""
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import shapely

from typing import List

from scipy import stats
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
import seaborn as sns

import cartopy
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from scripts.drafts.eng_utils import drop_nans_and_flatten
from scripts.drafts.eng_utils import (
    calculate_monthly_mean,
    calculate_spatial_mean,
    create_double_year,
)
from scripts.drafts.eng_utils import get_unmasked_data
from scripts.drafts.eng_utils import get_non_coord_variables
from scripts.drafts.eng_utils import caclulate_std_of_mthly_seasonality
from scripts.drafts.eng_utils import select_pixel, turn_tuple_to_point

# ------------------------------------------------------------------------------
# Histograms (Marginal Distributions)
# ------------------------------------------------------------------------------


def plot_marginal_distribution(
    dataArray, color, ax=None, title="", xlabel="DEFAULT", summary=False, **kwargs
):
    """ """
    # if no ax create one
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    # flatten the dataArray
    da_flat = drop_nans_and_flatten(dataArray)
    if summary:
        min, max, mean, median = (
            da_flat.min(),
            da_flat.max(),
            da_flat.mean(),
            np.median(da_flat),
        )

    # plot the histogram
    sns.distplot(da_flat, ax=ax, color=color, **kwargs)
    warnings.warn(
        "Hardcoding the values of the units becuase they should have already been converted to mm day-1"
    )

    if title == "":
        if summary:
            title = f"Density Plot of {dataArray.name} \nmin: {min:.2f} max: {max:.2f} mean: {mean:.2f} median: {median:.2f} "
        else:
            title = f"Density Plot of {dataArray.name}"

    ax.set_title(title)

    if xlabel == "DEFAULT":
        xlabel = f"Mean Monthly {dataArray.name} [mm day-1]"

    ax.set_xlabel(xlabel)

    return ax


# ------------------------------------------------------------------------------
# Joint Distributions
# ------------------------------------------------------------------------------


def plot_hexbin_comparisons(da1, da2, bins=None, mincnt=0.5, title_extra=None):
    """
    Arguments:
    ---------
    : bins (str, int, list, None)
        The binning of the colors for the histogram.
        Can be 'log', None, an integer for dividing into number of bins.
        If a list then used to define the lower bound of the bins to be used
    : mincnt (int, float)
        The minimum count for a color to be shown
    """
    data_array1 = drop_nans_and_flatten(da1)
    data_array2 = drop_nans_and_flatten(da2)

    var_dataset_x = data_array1
    var_dataset_y = data_array2
    r_value = pearsonr(data_array1, data_array2)

    fig, ax = plt.subplots(figsize=(12, 8))

    # plot the data
    hb = ax.hexbin(var_dataset_x, var_dataset_y, bins=bins, gridsize=40, mincnt=mincnt)

    # draw the 1:1 line (showing datasets exactly the same)
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3", label="1:1")

    # axes options
    dataset_name_x = da1.name.split("_")[0]
    dataset_name_y = da2.name.split("_")[0]
    title = f"Evapotranspiration: {dataset_name_x} vs. {dataset_name_y} \n Pearsons R: {r_value[0]:.2f} \n {title_extra}"

    ax.set_xlabel(dataset_name_x)
    ax.set_ylabel(dataset_name_y)
    ax.set_title(title)

    # colorbar
    cb = fig.colorbar(hb, ax=ax)
    if bins == "log":
        cb.set_label("log10(counts)")
    else:
        cb.set_label("counts")

    title = (
        f"{title_extra}{dataset_name_x}_v_{dataset_name_y}{bins}_{mincnt}_hexbin.png"
    )
    return fig, title


def hexbin_jointplot_sns(
    d1, d2, col1, col2, bins="log", mincnt=0.5, xlabel="", ylabel=""
):
    """
    Arguments:
    ---------
    : da1 (np.ndarray)
        numpy array of data (should be same lengths!)
    : da2 (np.ndarray)
        numpy array of data (should be same lengths!)
    : col1 (tuple)
        seaborn color code as tuple (rgba) e.g. `sns.color_palette()[0]`
    : col2 (tuple)
        seaborn color code as tuple (rgba) e.g. `sns.color_palette()[0]`
    : bins (str,list,None)
        how to bin your variables. If str then should be 'log'
    : mincnt (int, float)
        the minimum count for a value to be shown on the plot
    """
    # assert False, "Need to implement a colorbar and fix the colorbar values for all products (shouldn't matter too much because all products now have teh exact same number of pixels)"
    jp = sns.jointplot(d1, d2, kind="hex", joint_kws=dict(bins=bins, mincnt=mincnt))
    jp.annotate(stats.pearsonr)

    # plot the 1:1 line
    ax = jp.ax_joint
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3", label="1:1")

    # color the marginal distributions separately
    for patch in jp.ax_marg_x.patches:
        patch.set_facecolor(col1)

    for patch in jp.ax_marg_y.patches:
        patch.set_facecolor(col2)

    # label the axes appropriately
    jp.ax_joint.set_xlabel(xlabel)
    jp.ax_joint.set_ylabel(ylabel)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return jp


# ------------------------------------------------------------------------------
# Temporal Plots / timeseries
# ------------------------------------------------------------------------------


def plot_pixel_tseries(da, loc, ax, map_plot=False, marker=False):
    """ (lat, lon) = (y, x) """
    pixel_da = select_pixel(da, loc)

    if marker:
        pixel_da.plot.line(ax=ax, marker="o")
    else:
        pixel_da.plot.line(ax=ax)
    # TODO: how to set the labels to months
    # import calendar
    # ax.set_xticklabels([m for m in calendar.month_abbr if m != ''])
    # ax.grid(True)

    # plot a
    if map_plot:
        # get the whole domain from the regions
        from engineering.regions import regions

        region = regions[0]
        # plot an inset map
        fig = plt.gcf()
        point = turn_tuple_to_point(loc)
        # ax2 = plot_inset_map2(ax, region, borders=True)
        ax2 = plot_inset_map(
            ax, region, borders=True, lakes=True, plot_point=True, point=point
        )
        add_point_location_to_map(point, ax2, **{"color": "black"})
        # print(type(ax2))

    return ax


def plot_mean_and_std(mean_ds, std_ds, ax):
    """
    Assumes the labels in the std dataset are just the same with '_std' added as suffix!
    """
    # TODO: remove this complexity and force datasets to have a TIME dimension
    # check the time coordinates (can be 'time', 'month', 'season' etc.)
    time_coords = [
        coord
        for coord in mean_ds.coords
        if coord not in ["lat", "lon", "latitude", "longitude", "x", "y"]
    ]
    time_coord = "time" if "time" in time_coords else time_coords[0]

    assert all(
        mean_ds[time_coord].values == std_ds[time_coord].values
    ), f"Both datasets should be on the same timesteps! \nCurrently: mean_ds min: {mean_ds.time.min()} max: {mean_ds.time.max()} vs. std_ds min: {std_ds.time.min()} max: {std_ds.time.max()}"

    mean_ds_vars = get_non_coord_variables(mean_ds)
    std_ds_vars = get_non_coord_variables(std_ds)
    time = mean_ds[time_coord].values
    # mean_var, std_var = mean_ds_vars[0], std_ds_vars[0]

    for ix, mean_var in enumerate(
        mean_ds_vars
    ):  # mean_var, std_var in zip(mean_ds_vars,std_ds_vars):
        std_var = std_ds_vars[ix]
        # TODO: implement colors
        # color = colors[ix]

        mean_ts = mean_ds[mean_var].values
        std_ts = std_ds[std_var].values
        max_y = mean_ts + std_ts
        min_y = mean_ts - std_ts
        # ax.plot(x=time, y=mean_ts)
        pd.DataFrame({mean_var: mean_ts}).plot.line(ax=ax, marker="o")
        ax.fill_between(time - 1, min_y, max_y, alpha=0.3)

    return ax


def plot_seasonality(ds, ax=None, ylabel=None, double_year=False, variance=False):
    """Plot the monthly seasonality of the dataset

    Arguments:
    ---------
    : ds (xr.Dataset)
    : ylabel (str, None)
        what is the y-axis label?
    : double_year (bool)
        Do you want to view 2 seasonal cycles to better visualise winter months
    : variance (bool)
        Do you want +- 1SD 'intervals' plotted?

    TODO: explore seaborn to see if there is a better way to plot uncertainties.
          Main Q: what is the format data must be in to get uncertainties of
           +- 1SD?
    """
    mthly_ds = calculate_monthly_mean(ds)
    seasonality = calculate_spatial_mean(mthly_ds)

    if double_year:
        seasonality = create_double_year(seasonality)

    if variance:
        seasonality_std = caclulate_std_of_mthly_seasonality(
            ds, double_year=double_year
        )
        # merge into one dataset
        # seasonality = xr.merge([seasonality,seasonality_std])

    if ax == None:
        fig, ax = plt.subplots(figsize=(12, 8))

    if variance:
        plot_mean_and_std(seasonality, seasonality_std, ax)
    else:
        seasonality.to_dataframe().plot.line(ax=ax, marker="o")
        ax.set_title("Spatial Mean Seasonal Time Series")
        plt.legend()

    if double_year:
        # add vertical line separator to distinguish between years
        ax.axvline(12, color="black", linestyle=":", alpha=0.5)
    if ylabel != None:
        ax.set_ylabel(ylabel)
    fig = plt.gcf()
    return fig, ax


def plot_normalised_seasonality(ds, double_year=False, variance=False):
    """ Normalise the seasonality by each months contribution to the annual mean total.

    Arguments:
    ---------
    : ds (xr.Dataset)
        the dataset to calculate the seasonality from
    : double_year (bool)
        if True then show two annual cycles to get a better picture of the
         seasonality.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    mthly_ds = calculate_monthly_mean(ds)
    norm_seasonality = mthly_ds.apply(lambda x: (x / x.sum(dim="month")) * 100)
    norm_seasonality = calculate_spatial_mean(norm_seasonality)

    if double_year:
        norm_seasonality = create_double_year(norm_seasonality)

    if variance:
        seasonality_std = caclulate_std_of_mthly_seasonality(
            ds, double_year=double_year
        )
        norm_seasonality_std = seasonality_std.apply(
            lambda x: (x / x.sum(dim="month")) * 100
        )
        # merge into one dataset
        # seasonality = xr.merge([seasonality,seasonality_std])

    fig, ax = plt.subplots(figsize=(12, 8))

    if variance:
        plot_mean_and_std(norm_seasonality, norm_seasonality_std, ax)
    else:
        norm_seasonality.to_dataframe().plot(ax=ax)
        ax.set_title("Spatial Mean Seasonal Time Series")
        plt.legend()

    if double_year:
        # add vertical line separator to distinguish between years
        ax.axvline(12, color="black", linestyle=":", alpha=0.5)

    # convert to dataframe (useful for plotting values)
    # norm_seasonality.to_dataframe().plot(ax=ax)
    ax.set_title("Normalised Seasonality")
    ax.set_ylabel("Contribution of month to annual total (%)")
    plt.legend()

    return fig


# ------------------------------------------------------------------------------
# Spatio-temporal plots
# ------------------------------------------------------------------------------


def plot_seasonal_spatial_means(seasonal_da, ax=None, **kwargs):
    """ for a given seasonal xarray object plot the 4 seasons spatial means"""
    assert "season" in [
        key for key in seasonal_da.coords.keys()
    ], f"'season' should be a coordinate in the seasonal_da object for using this plotting functionality. \n Currently: {[key for key in seasonal_da.coords.keys()]}"
    assert isinstance(
        seasonal_da, xr.DataArray
    ), f"seasonal_da should be of type: xr.DataArray. Currently: {type(seasonal_da)}"
    scale = 1
    if ax == None:
        fig, axs = plt.subplots(2, 2, figsize=(12 * scale, 8 * scale))
    try:
        var = seasonal_da.name
    except:
        assert False, "sesaonal_da needs to be named!"
    for i in range(4):
        ax = axs[np.unravel_index(i, (2, 2))]
        seasonal_da.isel(season=i).plot(ax=ax, **kwargs)
        season_str = str(seasonal_da.isel(season=i).season.values)
        ax.set_title(f"{var} {season_str}")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


# ------------------------------------------------------------------------------
# Spatial Plots
# ------------------------------------------------------------------------------


def plot_masked_spatial_and_hist(
    dataMask, DataArrays, colors, titles, scale=1.5, **kwargs
):
    """ SPATIAL and HISTOGRAM plots to show the conditional distributions given
         a particular mask.

    Arguments:
    ---------
    : dataMask (xr.DataArray)
        Mask for a particular area
    : DataArrays (list, tuple, iterable?)
        list of xr.DataArrays to use for the data.
    """
    assert all(
        [isinstance(da, xr.DataArray) for da in DataArrays]
    ), f"Currently only works when every member of DataArrays are xr.DataArray. Currently: {[type(da) for da in DataArrays]}"
    assert len(colors) == len(
        DataArrays
    ), f"Len of the colors has to be equal to the len of the DataArrays \n Currently len(colors): {len(colors)} \tlen(DataArrays): {len(DataArrays)}"
    assert len(titles) == len(
        DataArrays
    ), f"Len of the titles has to be equal to the len of the DataArrays \n Currently len(titles): {len(titles)} \tlen(DataArrays): {len(DataArrays)}"

    fig, axs = plt.subplots(2, len(DataArrays), figsize=(12 * scale, 8 * scale))
    for j, DataArray in enumerate(DataArrays):
        if "time" in DataArray.dims:
            # if time variable e.g. Evapotranspiration
            dataArray = get_unmasked_data(DataArray.mean(dim="time"), dataMask)
        else:
            # if time constant e.g. landcover
            dataArray = get_unmasked_data(DataArray, dataMask)

        # get the axes for the spatial plots and the histograms
        ax_map = axs[0, j]
        ax_hist = axs[1, j]
        color = colors[j]
        title = titles[j]

        ax_map.set_title(f"{dataArray.name}")
        ylim = [0, 1.1]
        xlim = [0, 7]
        ax_hist.set_ylim(ylim)
        ax_hist.set_xlim(xlim)

        # plot the map
        plot_mean_time(dataArray, ax_map, add_colorbar=True, **kwargs)
        # plot the histogram
        plot_marginal_distribution(
            dataArray, color, ax=ax_hist, title=None, xlabel=dataArray.name
        )
        # plot_masked_histogram(ax_hist, dataArray, color, dataset)

    return fig


def plot_xarray_on_map(
    da, borders=True, coastlines=True, lakes=False, ax=None, cbar=None, **kwargs
):
    """ Plot the LOCATION of an xarray object """
    # get the center points for the maps
    mid_lat = np.mean(da.lat.values)
    mid_lon = np.mean(da.lon.values)

    # create the base layer
    if ax is None:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(
            1, 1, 1, projection=cartopy.crs.Orthographic(mid_lon, mid_lat)
        )
    # ax = plt.axes(projection=cartopy.crs.Orthographic(mid_lon, mid_lat))

    vmin = kwargs.pop("vmin", None)
    vmax = kwargs.pop("vmax", None)
    if cbar is None:
        da.plot(ax=ax, transform=cartopy.crs.PlateCarree(), vmin=vmin, vmax=vmax)
    else:
        # have to make 2D!
        da = da.squeeze(dim="time", drop=True)
        im = da.plot.pcolormesh(ax=ax, add_colorbar=False, vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(im)

    ax.coastlines()
    ax.add_feature(cartopy.feature.BORDERS, linestyle=":")
    if lakes:
        ax.add_feature(cartopy.feature.LAKES, facecolor=None)
    fig = plt.gcf()
    ax.outline_patch.set_visible(False)

    return fig, ax, cbar


def get_river_features():
    """ Get the 10m river features from NaturalEarth and turn into shapely.geom
    Note: https://github.com/SciTools/cartopy/issues/945

    """
    shp_path = cartopy.io.shapereader.natural_earth(
        resolution="10m", category="physical", name="rivers_lake_centerlines"
    )

    water_color = "#3690f7"
    shp_contents = cartopy.io.shapereader.Reader(shp_path)
    river_generator = shp_contents.geometries()
    river_feature = cartopy.feature.ShapelyFeature(
        river_generator,
        cartopy.crs.PlateCarree(),
        edgecolor=water_color,
        facecolor="none",
    )

    return river_feature


def plot_sub_geolocation(region, ax, lakes=False, borders=False, rivers=False):
    """ to be plot using axins methods

    https://matplotlib.org/gallery/axes_grid1/inset_locator_demo.html
    """
    lonmin, lonmax, latmin, latmax = (
        region.lonmin,
        region.lonmax,
        region.latmin,
        region.latmax,
    )
    ax.add_feature(cartopy.feature.COASTLINE)
    if borders:
        ax.add_feature(cartopy.feature.BORDERS, linestyle=":")
    if lakes:
        ax.add_feature(cartopy.feature.LAKES)
    if rivers:
        # assert False, "Rivers are not yet working in this function"
        river_feature = get_river_features()
        ax.add_feature(river_feature)
    return


def plot_geog_location(region, lakes=False, borders=False, rivers=False, scale=1):
    """ use cartopy to plot the region (defined as a namedtuple object)

    Arguments:
    ---------
    : region (Region namedtuple)
        region of interest bounding box defined in engineering/regions.py
    : lakes (bool)
        show lake features
    : borders (bool)
        show lake features
    : rivers (bool)
        show river features (@10m scale from NaturalEarth)
    """
    lonmin, lonmax, latmin, latmax = (
        region.lonmin,
        region.lonmax,
        region.latmin,
        region.latmax,
    )
    figsize = (12 * scale, 8 * scale)
    fig = plt.figure(figsize=figsize)
    ax = fig.gca(projection=cartopy.crs.PlateCarree())
    ax.add_feature(cartopy.feature.COASTLINE)
    if borders:
        ax.add_feature(cartopy.feature.BORDERS, linestyle=":")
    if lakes:
        ax.add_feature(cartopy.feature.LAKES)
    if rivers:
        # assert False, "Rivers are not yet working in this function"
        river_feature = get_river_features()
        ax.add_feature(river_feature)

    ax.set_extent([lonmin, lonmax, latmin, latmax])

    # plot the lat lon labels
    # https://scitools.org.uk/cartopy/docs/v0.15/examples/tick_labels.html
    # https://stackoverflow.com/questions/49956355/adding-gridlines-using-cartopy
    xticks = np.linspace(lonmin, lonmax, 5)
    yticks = np.linspace(latmin, latmax, 5)

    ax.set_xticks(xticks, crs=cartopy.crs.PlateCarree())
    ax.set_yticks(yticks, crs=cartopy.crs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    fig = plt.gcf()

    return fig, ax


def add_point_location_to_map(point, ax, color=(0, 0, 0, 1), **kwargs):
    """ """
    assert isinstance(
        point, shapely.geometry.point.Point
    ), f"point should be of type shapely.geometry.point.Point. Currently: {type(point)}"
    # assert isinstance(ax, cartopy.mpl.geoaxes.GeoAxesSubplot), f"Axes need to be cartopy.mpl.geoaxes.GeoAxesSubplot. Currently: {type(ax)}"
    ax.scatter(
        point.x, point.y, transform=cartopy.crs.PlateCarree(), c=[color], **kwargs
    )

    return


def add_points_to_map(ax, geodf, point_colors="#0037ff"):
    """ Add the point data stored in `geodf.geometry` as points to ax
    Arguments:
    ---------
    : geodf (geopandas.GeoDataFrame)
        gpd.GeoDataFrame with a `geometry` column containing shapely.Point geoms
    : ax (cartopy.mpl.geoaxes.GeoAxesSubplot)
    """
    assert isinstance(
        ax, cartopy.mpl.geoaxes.GeoAxesSubplot
    ), f"Axes need to be cartopy.mpl.geoaxes.GeoAxesSubplot. Currently: {type(ax)}"
    points = geodf.geometry.values

    # [add_point_location_to_map(point, ax, color="0037ff") for point in points]
    ax.scatter(
        [point.x for point in points],
        [point.y for point in points],
        transform=cartopy.crs.PlateCarree(),
        color=point_colors,
    )

    return ax


def plot_stations_on_region_map(
    region, station_location_df, point_colors="#0037ff", scale=1
):
    """ Plot the station locations in `station_location_df` on a map of the region

    Arguments:
    ---------
    : region (Region, namedtuple)
    : station_location_df (geopandas.GeoDataFrame)
        gpd.GeoDataFrame with a `geometry` column containing shapely.Point geoms

    Returns:
    -------
    : fig (matplotlib.figure.Figure)
    : ax (cartopy.mpl.geoaxes.GeoAxesSubplot)
    """
    fig, ax = plot_geog_location(
        region, lakes=True, borders=True, rivers=True, scale=scale
    )
    ax = add_points_to_map(ax, station_location_df, point_colors=point_colors)

    return fig, ax


def add_sub_region_box(ax, subregion, color):
    """ Plot a box for the subregion on the cartopy axes.
    TODO: implement a check where the subregion HAS TO BE inside the axes limits

    Arguments:
    ---------
    : ax (cartopy.mpl.geoaxes.GeoAxesSubplot)
        axes that you are plotting on
    : subregion (Region namedtuple)
        region of interest bounding box (namedtuple)
    """
    geom = geometry.box(
        minx=subregion.lonmin,
        maxx=subregion.lonmax,
        miny=subregion.latmin,
        maxy=subregion.latmax,
    )
    ax.add_geometries([geom], crs=cartopy.crs.PlateCarree(), color=color, alpha=0.3)
    return ax


def plot_inset_map(
    ax,
    region,
    borders=False,
    lakes=False,
    rivers=False,
    plot_point=False,
    point=None,
    width="40%",
    height="40%",
    loc="upper right",
):
    """ """
    axins = inset_axes(
        ax,
        width=width,
        height=height,
        loc=loc,
        axes_class=cartopy.mpl.geoaxes.GeoAxes,
        axes_kwargs=dict(map_projection=cartopy.crs.PlateCarree()),
    )
    axins.tick_params(labelleft=False, labelbottom=False)

    # plot the region
    lonmin, lonmax, latmin, latmax = (
        region.lonmin,
        region.lonmax,
        region.latmin,
        region.latmax,
    )
    axins.add_feature(cartopy.feature.COASTLINE)
    if borders:
        axins.add_feature(cartopy.feature.BORDERS, linestyle=":")
    if lakes:
        axins.add_feature(cartopy.feature.LAKES)
    if rivers:
        river_feature = get_river_features()
        axins.add_feature(river_feature)
    if plot_point:
        assert point != None
        axins.scatter(point.x, point.y, transform=cartopy.crs.PlateCarree(), c="black")
    axins.set_extent([lonmin, lonmax, latmin, latmax])

    return axins
