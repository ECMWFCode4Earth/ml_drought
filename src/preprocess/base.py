from pathlib import Path
import xarray as xr
import numpy as np
import pandas as pd

from typing import List, Optional

from ..utils import Region, region_lookup
from .utils import select_bounding_box

__all__ = ['BasePreProcessor', 'Region']

xesmf = None


class BasePreProcessor:
    """Base for all pre-processor classes. The preprocessing classes
    are responsible for taking the raw data exports and normalizing them
    so that they can be ingested by the feature engineering class.
    This involves:
    - subsetting the ROI (default is Kenya)
    - regridding to a consistent spatial grid (pixel size / resolution)
    - resampling to a consistent time step (hourly, daily, monthly)
    - assigning coordinates to `.nc` files (latitude, longitude, time)

    Attributes:
    ----------
    data_folder: Path, default: Path('data')
        The location of the data folder.
    """
    dataset: str
    static: bool = False
    analysis: bool = False

    def __init__(self, data_folder: Path = Path('data'),
                 output_name: Optional[str] = None) -> None:

        global xesmf
        if xesmf is None:
            import xesmf
        self.data_folder = data_folder
        self.raw_folder = self.data_folder / 'raw'
        self.preprocessed_folder = self.data_folder / 'interim'

        if not self.preprocessed_folder.exists():
            self.preprocessed_folder.mkdir(exist_ok=True, parents=True)

        try:
            if output_name is None:
                output_name = self.dataset

            if self.static:
                folder_prefix = f'static/{output_name}'
            else:
                folder_prefix = output_name

            if self.analysis:
                self.out_dir = self.data_folder / 'analysis' / f'{folder_prefix}_preprocessed'
            else:
                self.out_dir = self.preprocessed_folder / f'{folder_prefix}_preprocessed'

            if not self.out_dir.exists():
                self.out_dir.mkdir(parents=True)

            self.interim = self.preprocessed_folder / f'{folder_prefix}_interim'
            if not self.interim.exists():
                self.interim.mkdir(parents=True)
        except AttributeError:
            print('A dataset attribute must be added for '
                  'the interim and out directories to be created')

    def get_filepaths(self, folder: str = 'raw') -> List[Path]:
        if folder == 'raw':
            target_folder = self.raw_folder / self.dataset
        else:
            target_folder = self.interim
        outfiles = list(target_folder.glob('**/*.nc'))
        outfiles.sort()
        return outfiles

    def regrid(self,
               ds: xr.Dataset,
               reference_ds: xr.Dataset,
               method: str = "nearest_s2d",
               reuse_weights: bool = False,
               selected_var: Optional[str] = None,
               clean: bool = True) -> xr.Dataset:
        """ Use xEMSF package to regrid ds to the same grid as reference_ds

        Arguments:
        ----------
        ds: xr.Dataset
            The dataset to be regridded
        reference_ds: xr.Dataset
            The reference dataset, onto which `ds` will be regridded
        method: str, {'bilinear', 'conservative', 'nearest_s2d', 'nearest_d2s', 'patch'}
            The method applied for the regridding
        selected_var: Optional[str]
            Option to select a single variable to be regridded (removes the others)
        """

        assert ('lat' in reference_ds.dims) & ('lon' in reference_ds.dims), \
            f'Need (lat,lon) in reference_ds dims Currently: {reference_ds.dims}'
        assert ('lat' in ds.dims) & ('lon' in ds.dims), \
            f'Need (lat,lon) in ds dims Currently: {ds.dims}'

        regridding_methods = ['bilinear', 'conservative', 'nearest_s2d', 'nearest_d2s', 'patch']
        assert method in regridding_methods, \
            f'{method} not an acceptable regridding method. Must be one of {regridding_methods}'

        # create the grid you want to convert TO (from reference_ds)
        ds_out = xr.Dataset(
            {'lat': (['lat'], reference_ds.lat),
             'lon': (['lon'], reference_ds.lon)}
        )

        shape_in = len(ds.lat), len(ds.lon)
        shape_out = len(reference_ds.lat), len(reference_ds.lon)
        # unique id so when parallel process doesn't write to same file
        uid = f"{np.random.rand(1)[0]:.2f}"

        # The weight file should be deleted by regridder.clean_weight_files(), but in case
        # something goes wrong and its not, lets use a descriptive filename
        if reuse_weights:
            # if not running in parallel can save time by reusing weights
            filename = f'{method}_{shape_in[0]}x{shape_in[1]}_\
            {shape_out[0]}x{shape_out[1]}.nc'.replace(' ', '')
        else:
            filename = f'{method}_{shape_in[0]}x{shape_in[1]}_\
            {shape_out[0]}x{shape_out[1]}_{uid}.nc'.replace(' ', '')
        savedir = self.preprocessed_folder / filename

        regridder = xesmf.Regridder(ds, ds_out, method,  # type: ignore
                                    filename=str(savedir),
                                    reuse_weights=False)

        variables = [v for v in ds.data_vars]
        if selected_var is not None:
            variables = [v for v in variables if v == selected_var]
        output_dict = {}
        for var in variables:
            print(f'- regridding var {var} -')
            output_dict[var] = regridder(ds[var])
        ds = xr.Dataset(output_dict)

        print(f'Regridded from {(regridder.Ny_in, regridder.Nx_in)} '
              f'to {(regridder.Ny_out, regridder.Nx_out)}')

        if clean:
            regridder.clean_weight_file()

        return ds

    @staticmethod
    def load_reference_grid(path_to_grid: Path) -> xr.Dataset:
        """Since the regridder only needs to the lat and lon values,
        there is no need to pass around an enormous grid for the regridding.

        In fact, only the latitude and longitude values are necessary!
        """
        full_dataset = xr.open_dataset(path_to_grid)

        assert {'lat', 'lon'} <= set(full_dataset.dims), \
            'Dimensions named lat and lon must be in the reference grid'
        return full_dataset[['lat', 'lon']]

    @staticmethod
    def resample_time(ds: xr.Dataset,
                      resample_length: str = 'M',
                      upsampling: bool = False,
                      time_coord: str = 'time') -> xr.Dataset:

        # TODO: would be nice to programmatically get upsampling / not
        ds = ds.sortby(time_coord)

        resampler = ds.resample({time_coord: resample_length})

        if not upsampling:
            return resampler.mean()
        else:
            return resampler.nearest()

    @staticmethod
    def chop_roi(ds: xr.Dataset,
                 subset_str: Optional[str] = 'kenya',
                 inverse_lat: bool = False) -> xr.Dataset:
        """ lookup the region information from the dictionary in
        `src.utils.region_lookup` and subset the `ds` object based on that
        region.
        """
        region = region_lookup[subset_str] if subset_str is not None else None
        if region is not None:
            ds = select_bounding_box(ds, region, inverse_lat=inverse_lat)

        return ds

    def merge_files(self, subset_str: Optional[str] = 'kenya',
                    resample_time: Optional[str] = 'M',
                    upsampling: bool = False,
                    filename: Optional[str] = None,
                    ignore_timesteps: Optional[List[str]] = None,
                    drop_invalid: bool = False,
                    nan_subset_str: Optional[str] = None,) -> None:
        """

        Arguments:
        ---------
        subset_str: Optional[str] = 'kenya'
            the name of the subset ()
        resample_time: Optional[str] = 'M',
            the period string used to resample the xr.Dataset object
            # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries
             .html#dateoffset-objects
        upsampling: bool = False,
            whether the timeseries is being UPSAMPLED (monthly -> daily)
        filename: Optional[str] = None,
            the name of the filename to save
        ignore_timesteps: Optional[List[str]] = None,
            manually remove invalid timesteps
        drop_invalid: bool = False,
            whether to automatically detect and remove invalid timesteps
        nan_subset_str: Optional[str] = None,
            the subset/Region object used to check for invalid timesteps
            i.e. does the data have non-nan values inside this box?
        """
        ds = xr.open_mfdataset(self.get_filepaths('interim'))

        # add the ability to manually ignore timesteps
        if ignore_timesteps is not None:
            ts = [pd.to_datetime(s) for s in ignore_timesteps]
            print(f'Removing timesteps: {ts} from the dataset before resampling')
            ds = ds.sel(
                time=[
                    t for t in ds.time.values
                    if not np.isin(pd.to_datetime(t), ts)
                ]
            )

        # automatically detect errored timesteps
        if drop_invalid:
            assert nan_subset_str is not None, 'Require a subset_str value' \
                'if automatically detecting errored timesteps!'
            variables = [v for v in ds.data_vars]
            assert len(variables) == 1, 'There should only be one variable for' \
                'automatic detection of errors.'
            variable = variables[0]
            ds = self._fix_invalid_timesteps(
                ds, variable=variable, subset_str=nan_subset_str
            )

        if resample_time is not None:
            ds = self.resample_time(ds, resample_time, upsampling)

        if filename is None:
            filename = f'data{"_" + subset_str if subset_str is not None else ""}.nc'
        out = self.out_dir / filename

        ds.to_netcdf(out)
        print(f"\n**** {out} Created! ****\n")


    @staticmethod
    def _fix_invalid_timesteps(ds: xr.Dataset,
                               variable: str,
                               subset_str: str = 'indian_ocean',
                               invert_lat: bool = True) -> xr.Dataset:
        """in the region_of_known_nans if the values are
        NON-NAN then that timestep has an error.

        We currently drop these values (safest). However, we
        are working on functionality to check for and invert
        latitudes.
        """
        region_of_known_nans = region_lookup[subset_str]

        latmin = region_of_known_nans.latmin
        latmax = region_of_known_nans.latmax
        lonmin = region_of_known_nans.lonmin
        lonmax = region_of_known_nans.lonmax

        # create a boolean DataArray of the timesteps that are ALL NAN
        # inside that bounding box.
        if invert_lat:
            boolean_da = (
                ds.sel(lat=slice(latmax, latmin), lon=slice(lonmin, lonmax))[variable]
                .isnull().mean(dim=['lat', 'lon']) == 1
            )
        else:
            boolean_da = (
                ds.sel(lat=slice(latmin, latmax), lon=slice(lonmin, lonmax))[variable]
                .isnull().mean(dim=['lat', 'lon']) == 1
            )

        invalid_times = ds.sel(time=~boolean_da).time.values
        if list(invalid_times) != []:
            print(f'Invalid timesteps for variable {variable} removed:')
            print(invalid_times)
        else:
            print(f'All timesteps for variable: {variable} valid! None removed.')

        out_ds = ds.sel(time=boolean_da)
        assert len(ds.time) != len(invalid_times), 'Not all timesteps should be '\
            'removed as invalid. This has two causes: 1) the Region ' \
            f'object: {subset_str} is not a valid ROI for your data ' \
            '2) you need to toggle the `invert_lat` parameter. Try 2) first.'

        return out_ds
