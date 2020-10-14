"""
- subset ROI
- merge into one time (~500MB)
"""
from pathlib import Path
from functools import partial
import xarray as xr
import multiprocessing
from shutil import rmtree
from typing import Optional
import pandas as pd

from .base import BasePreProcessor


class MantleModisPreprocessor(BasePreProcessor):
    """ Preprocesses the Mantle Modis data (from the AWS S3 .tif format)"""

    dataset = "mantle_modis"

    def _preprocess_single(
        self,
        netcdf_filepath: Path,
        subset_str: Optional[str] = "kenya",
        regrid: Optional[xr.Dataset] = None,
    ) -> None:
        """Run the Preprocessing steps for the CHIRPS data

        Process:
        -------
        * assign time stamp
        * assign lat lon
        * create new dataset with these dimensions
        * Save the output file to new folder
        """
        print(f"Starting work on {netcdf_filepath.name}")
        # 1. read in the dataset
        ds = xr.open_dataset(netcdf_filepath)

        # 2. assign time stamp! (in the filename)
        time = pd.to_datetime(netcdf_filepath.name.split("_")[0])
        ds = ds.assign_coords(time=time)
        ds = ds.expand_dims("time")

        # 3.. chop out ROI
        if subset_str is not None:
            try:
                ds = self.chop_roi(ds, subset_str)
            except AssertionError:
                ds = self.chop_roi(ds, subset_str, inverse_lat=True)

        if regrid is not None:
            try:
                ds = self.regrid(ds, regrid)
            except ImportError:
                # the environment doesn't have esmpy does it have cdo?
                print("Use the ESMPY Environment (problems with gdal)")

        # 6. create the filepath and save to that location
        assert (
            netcdf_filepath.name[-3:] == ".nc"
        ), f"filepath name should be a .nc file. Currently: {netcdf_filepath.name}"

        filename = self.create_filename(
            netcdf_filepath, subset_name=subset_str if subset_str is not None else None,
        )
        print(f"Saving to {self.interim}/{filename}")
        ds.to_netcdf(self.interim / filename)

        var = list(ds.data_vars)[0]
        print(f"** Done for {var} {netcdf_filepath.name} **")

    @staticmethod
    def create_filename(
        netcdf_filepath: Path, subset_name: Optional[str] = None
    ) -> str:
        """
        chirps-v2.0.2009.pentads.nc
            =>
        chirps-v2.0.2009.pentads_kenya.nc
        """
        if subset_name is not None:
            final_part = netcdf_filepath.name.replace("_vci", f"_{subset_name}")
        else:
            final_part = netcdf_filepath.name.replace("_vci", "")
        new_filename = f"{netcdf_filepath.parents[1].name}_{final_part}"
        return new_filename

    def preprocess(
        self,
        subset_str: Optional[str] = "kenya",
        regrid: Optional[Path] = None,
        resample_time: Optional[str] = "M",
        upsampling: bool = False,
        # parallel: bool = False,
        n_processes: int = 1,
        cleanup: bool = True,
    ) -> None:
        """ Preprocess all of the CHIRPS .nc files to produce
        one subset file.

        Arguments
        ----------
        subset_str: Optional[str] = 'kenya'
            Whether to subset Kenya when preprocessing
        regrid_path: Optional[Path] = None
            If a Path is passed, the CHIRPS files will be regridded to have the same
            grid as the dataset at that Path. If None, no regridding happens
        resample_time: str = 'M'
            If not None, defines the time length to which the data will be resampled
        upsampling: bool = False
            If true, tells the class the time-sampling will be upsampling. In this case,
            nearest instead of mean is used for the resampling
        n_processes: int = 1
            If > 1, run the preprocessing in parallel
        cleanup: bool = True
            If true, delete interim files created by the class
        """
        print(f"Reading data from {self.raw_folder}. Writing to {self.interim}")

        # get the filepaths for all of the downloaded data
        nc_files = self.get_filepaths()

        regrid_path = regrid
        if regrid_path is not None:
            regrid = self.load_reference_grid(regrid_path)
        else:
            regrid = None

        n_processes = max(n_processes, 1)
        if n_processes > 1:  #  PARALLEL
            pool = multiprocessing.Pool(processes=n_processes)

            outputs = pool.map(
                partial(self._preprocess_single, subset_str=subset_str, regrid=regrid,),
                nc_files,
            )
            print("\nOutputs (errors):\n\t", outputs)
        else:  #  SEQUENTIAL
            for file in nc_files:
                self._preprocess_single(file, subset_str, regrid)

        # merge all of the timesteps
        self.merge_files(subset_str, resample_time, upsampling)

        if cleanup:
            rmtree(self.interim)
