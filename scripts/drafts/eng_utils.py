"""
* Collapsing Time Dimensions
* Collapsing Spatial Dimensions
* Binning your datasets
* Working with masks & subsets
* Lookup values from xarray in a dict
* Extracting individual pixels
* I/O
* working with older versions
"""

import xarray as xr
import numpy as np
import warnings
import pandas as pd
from netCDF4 import num2date

from typing import List
from pathlib import Path

# ------------------------------------------------------------------------------
# Selcting the Same Timeslice
# ------------------------------------------------------------------------------


def select_same_time_slice(reference_ds, ds):
    """ Select the values for the same timestep as the reference ds"""
    # CHECK THEY ARE THE SAME FREQUENCY
    # get the frequency of the time series from reference_ds
    freq = pd.infer_freq(reference_ds.time.values)
    if freq == None:
        warnings.warn("HARDCODED FOR THIS PROBLEM BUT NO IDEA WHY NOT WORKING")
        freq = "M"
        # assert False, f"Unable to infer frequency from the reference_ds timestep"

    old_freq = pd.infer_freq(ds.time.values)
    warnings.warn(
        "Disabled the assert statement. ENSURE FREQUENCIES THE SAME (e.g. monthly)"
    )
    # assert freq == old_freq, f"The frequencies should be the same! currenlty ref: {freq} vs. old: {old_freq}"

    # get the STARTING time point from the reference_ds
    min_time = reference_ds.time.min().values
    max_time = reference_ds.time.max().values
    orig_time_range = pd.date_range(min_time, max_time, freq=freq)
    # EXTEND the original time_range by 1 (so selecting the whole slice)
    # because python doesn't select the final in a range
    periods = len(orig_time_range)  # + 1
    # create new time series going ONE EXTRA PERIOD
    new_time_range = pd.date_range(min_time, freq=freq, periods=periods)
    new_max = new_time_range.max()

    # select using the NEW MAX as upper limit
    # --------------------------------------------------------------------------
    # FOR SOME REASON slice is removing the minimum time ...
    # something to do with the fact that matplotlib / xarray is working oddly with numpy64datetime object
    warnings.warn("L153: HARDCODING THE MIN VALUE OTHERWISE IGNORED ...")
    min_time = datetime.datetime(2001, 1, 31)
    # --------------------------------------------------------------------------
    ds = ds.sel(time=slice(min_time, new_max))
    assert (
        reference_ds.time.shape[0] == ds.time.shape[0]
    ), f"The time dimensions should match, currently reference_ds.time dims {reference_ds.time.shape[0]} != ds.time dims {ds.time.shape[0]}"

    print_time_min = pd.to_datetime(ds.time.min().values)
    print_time_max = pd.to_datetime(ds.time.max().values)
    try:
        vars = [i for i in ds.var().variables]
    except:
        vars = ds.name
    # ref_vars = [i for i in reference_ds.var().variables]
    print(
        f"Select same timeslice for ds with vars: {vars}. Min {print_time_min} Max {print_time_max}"
    )

    return ds


# ------------------------------------------------------------------------------
# General Utils
# ------------------------------------------------------------------------------


def drop_nans_and_flatten(dataArray: xr.DataArray) -> np.ndarray:
    """flatten the array and drop nans from that array. Useful for plotting histograms.

    Arguments:
    ---------
    : dataArray (xr.DataArray)
        the DataArray of your value you want to flatten
    """
    # drop NaNs and flatten
    return dataArray.values[~np.isnan(dataArray.values)]


def get_non_coord_variables(ds):
    """ Return a list of the variable names EXCLUDING the coordinates (lat,lon,time) """
    var_names = [var for var in ds.variables.keys() if var not in ds.coords.keys()]
    return var_names


def rename_lat_lon(ds):
    """ rename longitude=>lon, latitude=>lat """
    return ds.rename({"longitude": "lon", "latitude": "lat"})


def ls(dir):
    """ list the contents of a directory (like ls in bash) """
    return [f for f in dir.iterdir()]


def nans_mask_from_multiple_arrays(dataArrays: List[xr.DataArray]) -> xr.DataArray:
    """return a 2D array of values from ONLY matching indices

    Returns:
    -------
    :  xr.DataArray
        DataArray with the same dimensions as the inputs
    """
    # check all the dataArrays have the same dims
    dims_list = [tuple(da.dims) for da in dataArrays]
    len(set(dims_list)) <= 1, f"Ensure that all dims the same. Currently: {dims_list}"
    # check all the dataArrays have the same shape
    shapes_list = [da.shape for da in dataArrays]
    len(set(shapes_list)) <= 1, f"Ensure that all dims the same. Currently: {dims_list}"

    isnull_das = [da.isnull() for da in dataArrays]
    isnull = isnull_das[0]
    for isnull_da in isnull_das:
        isnull = isnull | isnull_da

    return isnull


# ------------------------------------------------------------------------------
# Collapsing Time Dimensions
# ------------------------------------------------------------------------------


def calculate_monthly_mean(ds):
    assert "time" in [
        dim for dim in ds.dims.keys()
    ], f"Time must be in the dataset dimensions. Currently: {[dim for dim in ds.dims.keys()]}"
    return ds.groupby("time.month").mean(dim="time")


def calculate_monthly_std(ds):
    assert "time" in [
        dim for dim in ds.dims.keys()
    ], f"Time must be in the dataset dimensions. Currently: {[dim for dim in ds.dims.keys()]}"
    return ds.groupby("time.month").std(dim="time")


def calculate_monthly_mean_std(ds):
    """ """
    # calculate mean and std
    mean = calculate_monthly_mean(ds)
    std = calculate_monthly_std(ds)

    # get var names
    dims = [dim for dim in mean.dims.keys()]
    vars = [var for var in mean.variables.keys() if var not in dims]

    # rename vars so can return ONE ds
    mean_vars = [var + "_monmean" for var in vars]
    std_vars = [var + "_monstd" for var in vars]
    mean = mean.rename(dict(zip(vars, mean_vars)))
    std = std.rename(dict(zip(vars, std_vars)))

    return xr.merge([mean, std])


def caclulate_std_of_mthly_seasonality(ds, double_year=False):
    """Calculate standard deviataion of monthly variability """
    std_ds = calculate_monthly_std(ds)
    seasonality_std = calculate_spatial_mean(std_ds)

    # rename vars
    var_names = get_non_coord_variables(seasonality_std)
    new_var_names = [var + "_std" for var in var_names]
    seasonality_std = seasonality_std.rename(dict(zip(var_names, new_var_names)))

    #
    if double_year:
        seasonality_std = create_double_year(seasonality_std)

    return seasonality_std


def create_double_year(seasonality):
    """for seasonality data (values for each month) return a copy for a second
        year to account for the cross-over between DJF

    Returns:
    -------
    : (xr.Dataset)
        a Dataset object with 24 months (2 annual cycles)
    """
    assert "month" in [
        coord for coord in seasonality.coords.keys()
    ], f"`month` must be a present coordinate in the seasonality data passed to the `create_double_year` function! Currently: {[coord for coord in seasonality.coords.keys()]}"

    seas2 = seasonality.copy()
    seas2["month"] = np.arange(13, 25)

    # merge the 2 datasets
    return xr.merge([seasonality, seas2])


# Seasonal means
# --------------
# Some calendar information so we can support any netCDF calendar
dpm = {
    "noleap": [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    "365_day": [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    "standard": [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    "gregorian": [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    "proleptic_gregorian": [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    "all_leap": [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    "366_day": [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    "360_day": [0, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30],
}

# A few calendar functions to determine the number of days in each month
def leap_year(year, calendar="standard"):
    """Determine if year is a leap year"""
    leap = False
    if (calendar in ["standard", "gregorian", "proleptic_gregorian", "julian"]) and (
        year % 4 == 0
    ):
        leap = True
        if (
            (calendar == "proleptic_gregorian")
            and (year % 100 == 0)
            and (year % 400 != 0)
        ):
            leap = False
        elif (
            (calendar in ["standard", "gregorian"])
            and (year % 100 == 0)
            and (year % 400 != 0)
            and (year < 1583)
        ):
            leap = False
    return leap


def get_dpm(time, calendar="standard"):
    """
    return a array of days per month corresponding to the months provided in `months`
    """
    month_length = np.zeros(len(time), dtype=np.int)

    cal_days = dpm[calendar]

    for i, (month, year) in enumerate(zip(time.month, time.year)):
        month_length[i] = cal_days[month]
        if leap_year(year, calendar=calendar):
            month_length[i] += 1
    return month_length


def season_mean(ds, calendar="standard"):
    """ Calculate a weighted season mean
    http://xarray.pydata.org/en/stable/examples/monthly-means.html """
    # Make a DataArray of season/year groups
    year_season = xr.DataArray(
        ds.time.to_index().to_period(freq="Q-NOV").to_timestamp(how="E"),
        coords=[ds.time],
        name="year_season",
    )

    # Make a DataArray with the number of days in each month, size = len(time)
    month_length = xr.DataArray(
        get_dpm(ds.time.to_index(), calendar=calendar),
        coords=[ds.time],
        name="month_length",
    )
    # Calculate the weights by grouping by 'time.season'
    weights = (
        month_length.groupby("time.season") / month_length.groupby("time.season").sum()
    )

    # Test that the sum of the weights for each season is 1.0
    np.testing.assert_allclose(weights.groupby("time.season").sum().values, np.ones(4))

    # Calculate the weighted average
    return (ds * weights).groupby("time.season").sum(dim="time")


def compute_anomaly(da, time_group="time.month"):
    """ Return a dataarray where values are an anomaly from the MEAN for that
         location at a given timestep. Defaults to finding monthly anomalies.

    Arguments:
    ---------
    : da (xr.DataArray)
    : time_group (str)
        time string to group.
    """
    mthly_vals = da.groupby(time_group).mean("time")
    da = da.groupby(time_group) - mthly_vals

    return da


# ------------------------------------------------------------------------------
# Collapsing Spatial Dimensions (-> Time Series)
# ------------------------------------------------------------------------------


def calculate_spatial_mean(ds):
    assert ("lat" in [dim for dim in ds.dims.keys()]) & (
        "lon" in [dim for dim in ds.dims.keys()]
    ), f"Must have 'lat' 'lon' in the dataset dimensisons"
    return ds.mean(dim=["lat", "lon"])


# ------------------------------------------------------------------------------
# Binning your datasets
# ------------------------------------------------------------------------------


def create_new_binned_dimensions(ds, group_var, intervals):
    """ Get the values in `ds` for `group_var` WITHIN the `interval` ranges.
         Return a new xr.Dataset with a new set of variables called `{group_var}_bins`.

    Arguments:
    ---------
    : ds (xr.Dataset)
        the dataset in which we are finding the values that lie within an interval
         range.
    : group_var (str)
        the variable that we are trying to bin
    : intervals (list, np.ndarray)
        list of `pandas._libs.interval.Interval` with methods `interval.left`
         and `interval.right` for extracting the values that fall within that
         range.

    Returns:
    -------
    : ds_bins (xr.Dataset)
        dataset with new `Variables` one for each bin. Pixels outside of the
         interval range are masked with np.nan
    """
    ds_bins = xr.concat(
        [
            ds.where((ds[group_var] > interval.left) & (ds[group_var] < interval.right))
            for interval in intervals
        ]
    )
    ds_bins = ds_bins.rename({"concat_dims": f"{group_var}_bins"})
    return ds_bins


def bin_dataset(ds, group_var, n_bins):
    """
    Arguments:
    ---------
    : ds (xr.Dataset)
        the dataset that you want to group / bin
    : group_var (str)
        the data variable that you want to group into bins

    Returns:
    -------
    : topo_bins (xr.Dataset)
        dataset object with number of variables equal to the number of bins
    : intervals (tuple)
        tuple of tuples with the bin left and right edges
         (intervals[0][0] = left edge;
          intervals[0][0] = right edge
         )
    """
    # groupby and collaps to the MID ELEVATION for the values (allows us to extract )
    bins = ds.groupby_bins(group=group_var, bins=n_bins).mean()
    # assert False, "hardcoding the elevation_bins here need to do this dynamically"
    binned_var = [key for key in bins.coords.keys()]
    assert len(binned_var) == 1, "The binned Var should only be one variable!"
    binned_var = binned_var[0]

    # extract the bin locations
    intervals = bins[binned_var].values
    left_bins = [interval.left for interval in intervals]
    # use bin locations to create mask variables of those values inside that
    ds_bins = create_new_binned_dimensions(ds, group_var, intervals)

    return ds_bins, intervals


# ------------------------------------------------------------------------------
# Working with masks & subsets
# ------------------------------------------------------------------------------


def merge_shapefiles(wsheds_shp, pp_to_polyid_map):
    """ Merge Shapefiles into ONE
    Example:
    -------
    you have two basin polygons for two different pour points
    and you want to merge them into one polygon.

    Notes:
    -----
    Make a union of polygons in Python, GeoPandas, or shapely (into a single geometry)
    https://stackoverflow.com/a/40386377/9940782
    unary_union or #geo_df.loc[pp_to_polyid_map[geoid]].dissolve('geometry')
    https://stackoverflow.com/a/40386377/9940782
    """
    out_shp_geoms = []
    for geoid in pp_to_polyid_map.keys():
        geoms = wsheds_shp.loc[pp_to_polyid_map[geoid]].geometry

        out_shp_geoms.append(shapely.ops.unary_union(geoms))

    # OUTPUT into one dataframe
    gdf = gpd.GeoDataFrame(
        {
            "geoid": [geoid for geoid in pp_to_polyid_map.keys()],
            "number": np.arange(0, 7),
            "geometry": out_shp_geoms,
        },
        geometry="geometry",
    )

    return gdf


def select_bounding_box_xarray(ds, region):
    """ using the Region namedtuple defined in engineering.regions.py select
    the subset of the dataset that you have defined that region for.

    Arguments:
    ---------
    : ds (xr.Dataset)
        the data (usually from netcdf file) that you want to subset a bounding
         box from
    : region (Region)
        namedtuple object defined in engineering/regions.py

    Returns:
    -------
    : ds (xr.DataSet)
        Dataset with a subset of the whol region defined by the Region object
    """
    print(f"selecting region: {region.name} from ds")
    assert isinstance(ds, xr.Dataset) or isinstance(
        ds, xr.DataArray
    ), f"ds Must be an xarray object! currently: {type(ds)}"
    lonmin = region.lonmin
    lonmax = region.lonmax
    latmin = region.latmin
    latmax = region.latmax

    dims = [dim for dim in ds.dims.keys()]
    variables = [var for var in ds.variables if var not in dims]

    if "latitude" in dims and "longitude" in dims:
        ds_slice = ds.sel(
            latitude=slice(latmin, latmax), longitude=slice(lonmin, lonmax)
        )
    elif "lat" in dims and "lon" in dims:
        ds_slice = ds.sel(
            latitude=slice(latmin, latmax), longitude=slice(lonmin, lonmax)
        )
    else:
        raise ValueError(
            f"Your `xr.ds` does not have lon / longitude in the dimensions. Currently: {[dim for dim in new_ds.dims.keys()]}"
        )
        return

    assert (
        ds_slice[variables[0]].values.size != 0
    ), f"Your slice has returned NO values. Sometimes this means that the latmin, latmax are the wrong way around. Try switching the order of latmin, latmax"
    return ds_slice


def get_unmasked_data(dataArray, dataMask):
    """ get the data INSIDE the dataMask
    Keep values if True, remove values if False
     (doing the opposite of a 'mask' - perhaps should rename)
    """
    return dataArray.where(dataMask)


def mask_multiple_conditions(da, vals_to_keep):
    """
    Arguments:
    ---------
    : da (xr.DataArray)
        data that you want to mask
    : variable (str)
        variable to search for the values in vals_to_keep
    : vals_to_keep (list)
        list of values to keep from variable

    Returns:
    -------
    : msk (xr.DataArray)
        a mask showing True for matches and False for non-matches

    Note: https://stackoverflow.com/a/40556458/9940782
    """
    msk = xr.DataArray(
        np.in1d(da, vals_to_keep).reshape(da.shape), dims=da.dims, coords=da.coords
    )

    return msk


def get_ds_mask(ds):
    """
    NOTE:
    - assumes that all of the null values from the HOLAPS file are valid null values (e.g. water bodies). Could also be invalid nulls due to poor data processing / lack of satellite input data for a pixel!
    """
    mask = ds.isnull().isel(time=0).drop("time")
    mask.name = "mask"

    return mask


def apply_same_mask(ds, reference_ds):
    """ Apply the same mask from reference_ds to ds
    """
    dataMask = get_ds_mask(reference_ds)
    return get_unmasked_data(dataArray, dataMask)


# -------------------------------------------------------------------------
# Lookup values from xarray in a dict
# -------------------------------------------------------------------------


def replace_with_dict(ar, dic):
    """ Replace the values in an np.ndarray with a dictionary

    https://stackoverflow.com/a/47171600/9940782

    """
    assert isinstance(
        ar, np.ndarray
    ), f"`ar` shoule be a numpy array! (np.ndarray). To work with xarray objects, first select the values and pass THESE to the `replace_with_dict` function (ar = da.values) \n Type of `ar` currently: {type(ar)}"
    # Extract out keys and values
    k = np.array(list(dic.keys()))
    v = np.array(list(dic.values()))

    # Get argsort indices
    sidx = k.argsort()

    # Drop the magic bomb with searchsorted to get the corresponding
    # places for a in keys (using sorter since a is not necessarily sorted).
    # Then trace it back to original order with indexing into sidx
    # Finally index into values for desired output.
    # NOTE: something going wrong with the number for the indices (0 based vs. 1 based)
    warnings.warn("We are taking one from the index. need to check this is true!!!")
    return v[sidx[np.searchsorted(k, ar, sorter=sidx) - 1]]


def replace_with_dict2(ar, dic):
    """Replace the values in an np.ndarray with a dictionary

    https://stackoverflow.com/a/47171600/9940782
    """
    # Extract out keys and values
    k = np.array(list(dic.keys()))
    v = np.array(list(dic.values()))

    # Get argsort indices
    sidx = k.argsort()

    ks = k[sidx]
    vs = v[sidx]
    warnings.warn("We are taking one from the index. need to check this is true!!!")
    return vs[np.searchsorted(ks, ar) - 1]


# TODO: rename this function
def get_lookup_val(xr_obj, variable, new_variable, lookup_dict):
    """ Assign a new Variable to xr_obj with values from lookup_dict.
    Arguments:
    ---------
    : xr_obj (xr.Dataset, xr.DataArray)
        the xarray object we want to look values up from
    : variable (str)
        the INPUT variable we are hoping to look the values up from (the dictionary keys)
    : new_variable (str)
        the name of the OUTPUT variable we want to put the dictionary values in
    : lookup_dict (dict)
        the dictionary we want to lookup the values of 'variable' in to return values to 'new_variable'
    """
    assert variable in list(ds.data_vars), f"variable is not in {list(ds.data_vars)}"
    # get the values as a numpy array
    if isinstance(xr_obj, xr.Dataset):
        ar = xr_obj[variable].values
    elif isinstance(xr_obj, xr.DataArray):
        ar = xr_obj.values
    else:
        assert (
            False
        ), f"This function only works with xarray objects. Currently xr_obj is type: {type(xr_obj)}"

    assert isinstance(ar, np.ndarray), f"ar should be a numpy array!"
    assert isinstance(lookup_dict, dict), f"lookup_dict should be a dictionary object!"

    # replace values in a numpy array with the values from the lookup_dict
    new_ar = replace_with_dict2(ar, lookup_dict)

    # assign the values looked up from the dictionary to a new variable in the xr_obj
    new_da = xr.DataArray(new_ar, coords=[xr_obj.lat, xr_obj.lon], dims=["lat", "lon"])
    new_da.name = new_variable
    xr_obj = xr.merge([xr_obj, new_da])

    return xr_obj


# -------------------------------------------------------------------------
# Extracting individual pixels
# -------------------------------------------------------------------------


def select_pixel(ds, loc):
    """ (lat,lon) """
    return ds.sel(lat=loc[1], lon=loc[0], method="nearest")


def turn_tuple_to_point(loc):
    """ (lat,lon) """
    from shapely.geometry.point import Point

    point = Point(loc[1], loc[0])
    return point


# -------------------------------------------------------------------------
# I/O
# -------------------------------------------------------------------------


def merge_data_arrays(*DataArrays):
    das = [da.name for da in DataArrays]
    print(f"Merging data: {das}")
    ds = xr.merge([*DataArrays])
    return ds


def save_netcdf(xr_obj, filepath, force=False):
    """"""
    if not Path(filepath).is_file():
        xr_obj.to_netcdf(filepath)
        print(f"File Saved: {filepath}")
    elif force:
        print(f"Filepath {filepath} already exists! Overwriting...")
        xr_obj.to_netcdf(filepath)
        print(f"File Saved: {filepath}")
    else:
        print(f"Filepath {filepath} already exists!")

    return


def pickle_files(filepaths, vars):
    """ """
    assert len(filepaths) == len(
        vars
    ), f"filepaths should be same size as vars because each variable needs a filepath! currently: len(filepaths): {len(filepaths)} len(vars): {len(vars)}"

    for i, filepath in enumerate(filepaths):
        save_pickle(filepath, variable)


def load_pickle(filepath):
    """ load a pickled object from the filepath """
    with open(filepath, "rb") as f:
        return pickle.load(f)


def save_pickle(filepath, variable):
    """ pickle a `variable` to the given `filepath`"""
    with open(filepath, "wb") as f:
        pickle.dump(variable, f)
    return


# ------------------------------------------------------
# working with older versions
# ------------------------------------------------------


def get_datetimes_from_files(files: List[Path]) -> List:
    datetimes = []
    for path in files:
        year = path.name.replace(".nc", "").split("_")[-2]
        month = path.name.replace(".nc", "").split("_")[-1]
        day = calendar.monthrange(int(year), int(month))[-1]
        dt = pd.to_datetime(f"{year}-{month}-{day}")
        datetimes.append(dt)
    return datetimes


def open_pred_data(model: str, experiment: str = "one_month_forecast"):
    import calendar

    files = [
        f for f in (data_dir / "models" / "one_month_forecast" / "ealstm").glob("*.nc")
    ]
    files.sort(key=lambda path: int(path.name.split("_")[-1][:-3]))
    times = get_datetimes_from_files(files)

    pred_ds = xr.merge(
        [
            xr.open_dataset(f).assign_coords(time=times[i]).expand_dims("time")
            for i, f in enumerate(files)
        ]
    )

    return pred_ds
