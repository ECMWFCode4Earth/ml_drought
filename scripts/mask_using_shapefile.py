from rasterio import features
from affine import Affine
import xarray as xr
import numpy as np
import geopandas as gpd


def transform_from_latlon(lat, lon):
    """ input 1D array of lat / lon and output an Affine transformation

    Arguments:
    ---------
    : lat (list, np.array)
        list of lat values
    : lon (list, np.array)
        list of lon values
    """
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    trans = Affine.translation(lon[0], lat[0])
    scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
    return trans * scale


def rasterize(shapes, coords, latitude="lat", longitude="lon", fill=np.nan, **kwargs):
    """Rasterize a list of (geometry, fill_value) tuples onto the given
    xray coordinates. This only works for 1d latitude and longitude
    arrays.

    usage:
    -----
    1. read shapefile to geopandas.GeoDataFrame
          `states = gpd.read_file(shp_dir+shp_file)`
    2. encode the different shapefiles that capture those lat-lons as different
        numbers i.e. 0.0, 1.0 ... and otherwise np.nan
          `shapes = (zip(states.geometry, range(len(states))))`
    3. Assign this to a new coord in your original xarray.DataArray
          `ds['states'] = rasterize(shapes, ds.coords, longitude='X', latitude='Y')`

    arguments:
    ---------
    : shapes ()
    : coords ()
    : latitude (str)
        the label for the latitude coordinate in the xarray object
    : longitude (str)
        the label for the longitude coordinate in the xarray object
    : **kwargs (dict)
        passed to `rasterio.rasterize` function

    attrs:
    -----
    :transform (affine.Affine): how to translate from latlon to ...?
    :raster (numpy.ndarray): use rasterio.features.rasterize fill the values
      outside the .shp file with np.nan
    :spatial_coords (dict): dictionary of {"X":xr.DataArray, "Y":xr.DataArray()}
      with "X", "Y" as keys, and xr.DataArray as values

    returns:
    -------
    :(xr.DataArray): DataArray with `values` of nan for points outside shapefile
      and coords `Y` = latitude, 'X' = longitude.


    """
    transform = transform_from_latlon(coords[latitude], coords[longitude])
    out_shape = (len(coords[latitude]), len(coords[longitude]))
    raster = features.rasterize(
        shapes,
        out_shape=out_shape,
        fill=fill,
        transform=transform,
        dtype=float,
        **kwargs
    )
    spatial_coords = {latitude: coords[latitude], longitude: coords[longitude]}
    return xr.DataArray(raster, coords=spatial_coords, dims=(latitude, longitude))


def add_shape_coord_from_data_array(
    xr_da, shp_path, coord_name, longitude="lon", latitude="lat"
):
    """ Create a new coord for the xr_da indicating whether or not it
         is inside the shapefile

        Creates a new coord - "coord_name" which will have integer values
         used to subset xr_da for plotting / analysis/

        Arguments:
        ---------
        : xr_da (xr.DataArray)
            the xarray object that you want to mask
        : shp_path (str, Path)
            the path to the shapefile that you want to use to mask the xr_da
        : coord_name (str)
            the coord to create in your xarray object (supports multiple
             geometries in the shapefile)
        : latitude (str)
            the coordinate name for the latitude in the xr_da
        : longitude (str)
            the coordinate name for the longitude in the xr_da
        Usage:
        -----
        # create a new coordinate in the xarray object
        precip_da = add_shape_coord_from_data_array(precip_da, "shape.shp", "shape")

        # use a boolean index to mask out values not inside precip_da.awash
        shape_da = precip_da.where(precip_da.shape==0, other=np.nan)
    """
    # 1. read in shapefile
    shp_gpd = gpd.read_file(shp_path)

    # 2. create a list of tuples (shapely.geometry, id)
    #    this allows for many different polygons within a .shp file (e.g. States of US)
    shapes = [(shape, n) for n, shape in enumerate(shp_gpd.geometry)]

    # BUT WE HAVE NO DICTIONARY FOR WHICH VALUES ARE WHICH
    # TODO: implement a lookup
    # shp_column = "LEVEL6"

    # 3. create a new coord in the xr_da which will be set to the id in `shapes`
    xr_da[coord_name] = rasterize(
        shapes, xr_da.coords, longitude=longitude, latitude=latitude
    )

    return xr_da
