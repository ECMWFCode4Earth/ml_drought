from pathlib import Path
import xarray as xr
from xarray.core.coordinates import DataArrayCoordinates
import numpy as np
from typing import Tuple, List, Optional, Dict

from ..utils import Region

features = None
Affine = None
gpd = None
Polygon = None


def select_bounding_box(
    ds: xr.Dataset, region: Region, inverse_lat: bool = False, inverse_lon: bool = False
) -> xr.Dataset:
    """ using the Region namedtuple defined in engineering.regions.py select
    the subset of the dataset that you have defined that region for.

    Arguments:
    ---------
    : ds (xr.Dataset)
        the data (usually from netcdf file) that you want to subset a bounding
         box from
    : region (Region)
        namedtuple object defined in engineering/regions.py
    : inverse_lat (bool) = False
        Whether to inverse the minimum and maximum latitudes
    : inverse_lon (bool) = False
        Whether to inverse the minimum and maximum longitudes

    Returns:
    -------
    : ds (xr.DataSet)
        Dataset with a subset of the whol region defined by the Region object
    """
    # print(f"selecting region: {region.name} from ds")
    assert isinstance(ds, xr.Dataset) or isinstance(ds, xr.DataArray), (
        f"ds. " f"Must be an xarray object! currently: {type(ds)}"
    )

    dims = list(ds.dims.keys())
    variables = [var for var in ds.variables if var not in dims]

    latmin, latmax, lonmin, lonmax = (
        region.latmin,
        region.latmax,
        region.lonmin,
        region.lonmax,
    )

    if "latitude" in dims and "longitude" in dims:
        ds_slice = ds.sel(
            latitude=slice(latmax, latmin) if inverse_lat else slice(latmin, latmax),
            longitude=slice(lonmax, lonmin) if inverse_lon else slice(lonmin, lonmax),
        )
    elif "lat" in dims and "lon" in dims:
        ds_slice = ds.sel(
            lat=slice(latmax, latmin) if inverse_lat else slice(latmin, latmax),
            lon=slice(lonmax, lonmin) if inverse_lon else slice(lonmin, lonmax),
        )
    else:
        raise ValueError(
            f"Your `xr.ds` does not have lon / longitude in the "
            f"dimensions. Currently: {[dim for dim in ds.dims.keys()]}"
        )

    for variable in variables:
        assert ds_slice[variable].values.size != 0, (
            f"Your slice has returned NO values. "
            f"Sometimes this means that the latmin, latmax are the wrong way around. "
            f"Try switching the order of latmin, latmax"
        )
    return ds_slice


class SHPtoXarray:
    def __init__(self):
        print(
            "the SHPtoXarray functionality requires "
            "rasterio, Affine, geopandas and shapely"
        )

        global features
        if features is None:
            from rasterio import features

        global Affine
        if Affine is None:
            from affine import Affine

        global gpd
        if gpd is None:
            import geopandas as gpd

        global Polygon
        if Polygon is None:
            from shapely.geometry import Polygon

    @staticmethod
    def transform_from_latlon(
        lat: xr.DataArray, lon: xr.DataArray
    ) -> Affine:  # type: ignore
        """ input 1D array of lat / lon and output an Affine transformation
        """
        lat = np.asarray(lat)
        lon = np.asarray(lon)
        trans = Affine.translation(lon[0], lat[0])  # type: ignore
        scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])  # type: ignore
        return trans * scale

    def rasterize(
        self,
        shapes: List[Polygon],  # type: ignore
        coords: DataArrayCoordinates,
        variable_name: str,
        **kwargs,
    ):
        """Rasterize a list of (geometry, fill_value) tuples onto the given
        xarray coordinates. This only works for 1d latitude and longitude
        arrays.
        """
        fill = np.nan
        transform: Affine = self.transform_from_latlon(  # type: ignore
            coords["lat"], coords["lon"]
        )
        out_shape: Tuple = (len(coords["lat"]), len(coords["lon"]))
        raster: np.ndarray = features.rasterize(  # type: ignore
            shapes,
            out_shape=out_shape,
            fill=fill,
            transform=transform,
            dtype=float,
            **kwargs,
        )
        spatial_coords: Dict = {"lat": coords["lat"], "lon": coords["lon"]}
        dims = ["lat", "lon"]

        return xr.Dataset({variable_name: (dims, raster)}, coords=spatial_coords)

    def shapefile_to_xarray(
        self,
        da: xr.DataArray,
        shp_path: Path,
        var_name: str = "region",
        lookup_colname: Optional[str] = None,
    ) -> xr.Dataset:
        """ Create a new coord for the da indicating whether or not it
         is inside the shapefile

        Creates a new coord - "var_name" which will have integer values
         used to subset da for plotting / analysis

        Arguments:
        ---------
        :da: xr.DataArray
            the `DataArray` with the shape that we want to rasterize the
            shapefile onto.

        :shp_path: Path
            the path to the .shp file to be converted into a categorical
            xr.Dataset.

        :var_name: str = 'region'
            the variable name in the new output Dataset

        :lookup_colname: Optional[str] = None
            the column that defines the `values` in the lookup
            dictionary when defining the (e.g. Region names)

            e.g. 'DISTNAME' in this shapefile below
               DISTID   DISTNAME  geometry
            0   101.0    NAIROBI  POLYGON ((36. -1 ...
            1   201.0     KIAMBU  POLYGON ((36. -0.7 ...

        Returns:
        -------
        :xr.Dataset
            Dataset with metadata associated with the areas in the shapefile.
            Stored as `ds.attrs['keys']` & `ds.attrs['values']`

        TODO: add a add_all_cols_as_attrs() function
        """
        # 1. read in shapefile
        gdf = gpd.read_file(shp_path)  # type: ignore

        # allow the user to see the column headers
        if lookup_colname is None:
            print("lookup_colname MUST be provided (see error message below)")
            print(gdf.head())

        assert (
            lookup_colname in gdf.columns
        ), f"lookup_colname must be one of: {list(gdf.columns)}"

        # 2. create a list of tuples (shapely.geometry, id)
        # this allows for many different polygons within a .shp file
        # (e.g. Admin Regions of Kenya)
        shapes = [(shape, n) for n, shape in enumerate(gdf.geometry)]

        # 3. create a new variable set to the id in `shapes` (same shape as da)
        ds = self.rasterize(shapes=shapes, coords=da.coords, variable_name=var_name)
        values = [value for value in gdf[lookup_colname].to_list()]
        keys = [str(key) for key in gdf.index.to_list()]
        data_vals = ds[[d for d in ds.data_vars][0]].values
        unique_values = np.unique(data_vals[~np.isnan(data_vals)])
        unique_values = [str(int(v)) for v in unique_values]

        # Check for None in keys/values
        keys = [
            key
            for key, value in zip(keys, values)
            if (values is not None) & (keys is not None)
        ]
        values = [
            value
            for key, value in zip(keys, values)
            if (values is not None) & (keys is not None)
        ]
        # assign to attrs
        ds.attrs["keys"] = ", ".join(keys)
        ds.attrs["values"] = ", ".join(values)
        ds.attrs["unique_values"] = ", ".join(unique_values)

        if ds[var_name].isnull().mean() <= 0.01:
            print("NOTE: Only 1% of values overlap with shapes")
            print("Are you certain the subset or shapefile are the correct region?")

        return ds
