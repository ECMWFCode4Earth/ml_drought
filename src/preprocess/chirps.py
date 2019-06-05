"""
- subset Kenya
- merge into one time (~500MB)
"""
from pathlib import Path
from functools import partial
import xarray as xr
import multiprocessing
from shutil import rmtree
from typing import Optional

from .base import BasePreProcessor, get_kenya
from .utils import select_bounding_box


class CHIRPSPreprocesser(BasePreProcessor):
    """ Preprocesses the CHIRPS data """

    dataset = 'chirps'

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
        ds = xr.open_dataset(netcdf_filepath).rename({'longitude': 'lon', 'latitude': 'lat'})

        # 2. chop out EastAfrica
        if subset_kenya:
            kenya_region = get_kenya()
            ds = select_bounding_box(ds, kenya_region)

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

        print(f"** Done for CHIRPS {netcdf_filepath.name} **")

    def preprocess(self, subset_kenya: bool = True,
                   regrid: Optional[Path] = None,
                   resample_time: Optional[str] = 'M',
                   upsampling: bool = False,
                   parallel: bool = False,
                   cleanup: bool = True) -> None:
        """ Preprocess all of the CHIRPS .nc files to produce
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
        parallel: bool = True
            If true, run the preprocessing in parallel
        cleanup: bool = True
            If true, delete interim files created by the class
        """
        print(f'Reading data from {self.raw_folder}. Writing to {self.interim}')

        # get the filepaths for all of the downloaded data
        nc_files = self.get_filepaths()

        if regrid is not None:
            regrid = self.load_reference_grid(regrid)

        if parallel:
            pool = multiprocessing.Pool(processes=100)
            outputs = pool.map(partial(self._preprocess_single, subset_kenya=subset_kenya,
                                       regrid=regrid), nc_files)
            print("\nOutputs (errors):\n\t", outputs)
        else:
            for file in nc_files:
                self._preprocess_single(file, subset_kenya, regrid)

        # merge all of the timesteps
        self.merge_files(subset_kenya, resample_time, upsampling)

        if cleanup:
            rmtree(self.interim)
