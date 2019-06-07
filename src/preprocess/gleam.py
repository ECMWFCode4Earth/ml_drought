from pathlib import Path
import xarray as xr
import numpy as np
from shutil import rmtree
from typing import Dict, List, Optional, Tuple

from .base import BasePreProcessor, get_kenya
from .utils import select_bounding_box


class GLEAMPreprocessor(BasePreProcessor):
    """Preprocess the GLEAM data
    """

    dataset = 'gleam'

    @staticmethod
    def _swap_dims_and_filter(ds: xr.Dataset) -> xr.Dataset:

        ds = ds.drop_dims('bnds')

        expected_dims = ['time', 'lat', 'lon']

        if list(ds.dims) == expected_dims:
            return ds

        print('Transposing arrays')
        dataarrays: Dict[str, Tuple[List[str], np.ndarray]] = {}
        for variable in ds.data_vars:
            transposed = np.swapaxes(ds[variable].values, 1, 2)
            dataarrays[variable] = (expected_dims, transposed)

        return xr.Dataset(dataarrays,
                          coords={'lat': ds.lat.values,
                                  'lon': ds.lon.values,
                                  'time': ds.time.values})

    def _preprocess_single(self, netcdf_filepath: Path,
                           subset_kenya: bool = True,
                           regrid: Optional[xr.Dataset] = None) -> None:
        """Run the Preprocessing steps for the CHIRPS data

        Process:
        -------
        * assign time stamp
        * assign lat lon
        * create new dataset with these dimensions
        * Save the output file to new folder
        """
        print(f'Starting work on {netcdf_filepath.name}')
        # 1. read in the dataset
        ds = xr.open_dataset(netcdf_filepath)

        # remove the time bounds variable
        ds = self._swap_dims_and_filter(ds)

        # 2. chop out EastAfrica
        if subset_kenya:
            kenya_region = get_kenya()
            ds = select_bounding_box(ds, kenya_region, inverse_lat=True)

        if regrid is not None:
            ds = self.regrid(ds, regrid)

        # 6. create the filepath and save to that location
        assert netcdf_filepath.name[-3:] == '.nc', \
            f'filepath name should be a .nc file. Currently: {netcdf_filepath.name}'

        filename = self.create_filename(
            netcdf_filepath.name,
            subset_name='kenya' if subset_kenya else None
        )
        print(f"Saving to {self.interim}/{filename}")
        ds.to_netcdf(self.interim / filename)

        print(f"** Done for GLEAM {netcdf_filepath.name} **")

    @staticmethod
    def create_filename(netcdf_filename: str,
                        subset_name: Optional[str] = None) -> str:
        """
        monthly/E_1980_2018_GLEAM_v3.3a_MO.nc
            =>
        E_1980_2018_GLEAM_v3.3a_MO_kenya.nc
        """
        filename_stem = netcdf_filename[:-3]
        if subset_name is not None:
            new_filename = f'{filename_stem}_{subset_name}.nc'
        else:
            new_filename = f'{filename_stem}.nc'
        return new_filename

    def preprocess(self, subset_kenya: bool = True,
                   regrid: Optional[Path] = None,
                   resample_time: Optional[str] = 'M',
                   upsampling: bool = False,
                   cleanup: bool = True) -> None:
        """ Preprocess all of the GLEAM .nc files to produce
        one subset file.

        Arguments
        ----------
        subset_kenya: bool = True
            Whether to subset Kenya when preprocessing
        regrid: Optional[Path] = None
            If a Path is passed, the CHIRPS files will be regridded to have the same
            grid as the dataset at that Path. If None, no regridding happens
        resample_time: str = 'M'
            If not None, defines the time length to which the data will be resampled
        upsampling: bool = False
            If true, tells the class the time-sampling will be upsampling. In this case,
            nearest instead of mean is used for the resampling
        cleanup: bool = True
            If true, delete interim files created by the class
        """
        print(f'Reading data from {self.raw_folder}. Writing to {self.interim}')

        # get the filepaths for all of the downloaded data
        nc_files = self.get_filepaths()

        if regrid is not None:
            regrid = self.load_reference_grid(regrid)

        for file in nc_files:
            self._preprocess_single(file, subset_kenya, regrid)

        # merge all of the timesteps
        self.merge_files(subset_kenya, resample_time, upsampling)

        if cleanup:
            rmtree(self.interim)
