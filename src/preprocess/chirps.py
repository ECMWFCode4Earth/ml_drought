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

from .base import BasePreProcessor


class CHIRPSPreprocessor(BasePreProcessor):
    """ Preprocesses the CHIRPS data """

    dataset = "chirps"

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
        ds = xr.open_dataset(netcdf_filepath).rename(
            {"longitude": "lon", "latitude": "lat"}
        )

        # 2. chop out EastAfrica
        if subset_str is not None:
            ds = self.chop_roi(ds, subset_str)

        if regrid is not None:
            ds = self.regrid(ds, regrid)

        # 6. create the filepath and save to that location
        assert (
            netcdf_filepath.name[-3:] == ".nc"
        ), f"filepath name should be a .nc file. Currently: {netcdf_filepath.name}"

        filename = self.create_filename(
            netcdf_filepath.name,
            subset_name=subset_str if subset_str is not None else None,
        )
        print(f"Saving to {self.interim}/{filename}")
        ds.to_netcdf(self.interim / filename)

        print(f"** Done for CHIRPS {netcdf_filepath.name} **")

    @staticmethod
    def create_filename(netcdf_filepath: str, subset_name: Optional[str] = None) -> str:
        """
        chirps-v2.0.2009.pentads.nc
            =>
        chirps-v2.0.2009.pentads_kenya.nc
        """
        if netcdf_filepath[-3:] == ".nc":
            filename_stem = netcdf_filepath[:-3]
        else:
            filename_stem = netcdf_filepath

        if subset_name is not None:
            new_filename = f"{filename_stem}_{subset_name}.nc"
        else:
            new_filename = f"{filename_stem}.nc"
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
        regrid: Optional[Path] = None
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

        if regrid is not None:
            regrid = self.load_reference_grid(regrid)

        n_processes = max(n_processes, 1)
        if n_processes > 1:
            pool = multiprocessing.Pool(processes=n_processes)

            outputs = pool.map(
                partial(self._preprocess_single, subset_str=subset_str, regrid=regrid),
                nc_files,
            )
            print("\nOutputs (errors):\n\t", outputs)
        else:
            for file in nc_files:
                self._preprocess_single(file, subset_str, regrid)

        # merge all of the timesteps
        self.merge_files(subset_str, resample_time, upsampling)

        if cleanup:
            rmtree(self.interim)
