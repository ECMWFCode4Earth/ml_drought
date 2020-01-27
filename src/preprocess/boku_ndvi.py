from pathlib import Path
from shutil import rmtree
import xarray as xr
import os
import multiprocessing
from functools import partial
import re
from datetime import datetime
import pandas as pd
import numpy as np

from typing import cast, Optional, Tuple

from .base import BasePreProcessor
# from src.analysis import


class BokuNDVIPreprocessor(BasePreProcessor):
    resolution: str

    def __init__(
        self,
        data_folder: Path = Path("data"),
        output_name: Optional[str] = None,
        resolution: str = "1000",
    ):
        self.resolution = str(resolution)
        self.static = False

        if self.resolution == "1000":
            # 1km pixel
            self.dataset: str = "boku_ndvi_1000"
        elif self.resolution == "250":
            # 250m pixel
            self.dataset: str = "boku_ndvi_250"
        else:
            assert False, (
                "Must provide str resolution of 1000 or 250"
                f"Provided: {resolution} Type: {type(resolution)}"
            )

        super().__init__(data_folder, output_name)

    @staticmethod
    def create_filename(netcdf_filepath: str, subset_name: Optional[str] = None) -> str:
        """
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

    def _parse_time_from_filename(self, filename) -> datetime:
        """
        extract the datetime from filename.

        Example:
            MCD13A2.t200915.006.EAv1.1_km_10_days_NDVI.O1.nc
            the 15th Monday of 2009

        returns datetime object
        """
        # regex pattern (4 digits after '.t')
        year_pattern = re.compile(r".t\d{4}")
        # extract the year from the filename
        year = year_pattern.findall(filename)[0].split(".t")[-1]

        if self.resolution == '1000':
            # extract the week_number (ISO 8601 week)
            week_num = year_pattern.split(filename)[-1].split(".")[0]

            return datetime.strptime(f"{year}-{week_num}-Mon", "%G-%V-%a")

        elif self.resolution == '250':
            assert False, 'HAVENT DONE THIS YET'
            # extract the day_number (ISO 8601 week)
            week_num = year_pattern.split(filename)[-1].split(".")[0]

            return datetime.strptime(f"{year}-{week_num}-Mon", "%G-%V-%a")

        else:
            assert False, 'Only working with two resolutions 1000 / 250'

    def create_new_dataarray(
        self, ds: xr.Dataset, timestamp: pd.Timestamp
    ) -> xr.Dataset:
        variable = "boku_ndvi"
        assert (
            np.array(timestamp).size == 1
        ), "The function only currently works with SINGLE TIMESTEPS."

        da = xr.DataArray(
            [ds[variable].values],
            dims=["time", "lat", "lon"],
            coords={"lon": ds.lon, "lat": ds.lat, "time": [timestamp]},
        )
        da.name = variable
        return da.to_dataset()

    # def _convert_to_VCI(self,):
    #     """Convert the BOKU NDVI data to VCI data

    #     Justification:
    #     The BOKU NDVI data is in the range 0-255 which
    #     suggests there is a scaling issue. Since the VCI
    #     is an anomaly score the raw values of the NDVI doesn't
    #     matter. Therefore, it makes sense to preprocess to VCI.
    #     """

    def _preprocess_single(
        self,
        netcdf_filepath: Path,
        subset_str: Optional[str] = "kenya",
        regrid: Optional[xr.Dataset] = None,
    ) -> None:
        """Run the Preprocessing steps for the BOKU NDVI data

        Process:
        -------
        * assign time stamp
        * assign lat lon
        * create new dataset with these dimensions
        * Save the output file to new folder

        NOTE:
        * the values are currently in the range 1-255
        * need to transform them to NDVI values
        * but require information about how to do this mapping to -1:1
        """
        print(f"Starting work on {netcdf_filepath.name}")
        # 1. read in the dataset
        ds = xr.open_dataset(netcdf_filepath)

        # assign time stamp
        timestamp = pd.to_datetime(self._parse_time_from_filename(netcdf_filepath.name))
        ds = self.create_new_dataarray(ds, timestamp)

        # 2. chop out EastAfrica
        if subset_str is not None:
            ds = self.chop_roi(ds, subset_str)

        # 3. regrid
        if regrid is not None:
            ds = self.regrid(ds, regrid)

        # 4. create the filepath and save to that location
        assert (
            netcdf_filepath.name[-3:] == ".nc"
        ), f"filepath name should be a .nc file. Currently: {netcdf_filepath.name}"

        filename = self.create_filename(
            netcdf_filepath.name,
            subset_name=subset_str if subset_str is not None else None,
        )
        print(f"Saving to {self.interim}/{filename}")
        ds.to_netcdf(self.interim / filename)

        print(f"** Done for BOKU NDVI {netcdf_filepath.name} **")

    def preprocess(
        self,
        subset_str: Optional[str] = "kenya",
        regrid: Optional[Path] = None,
        resample_time: Optional[str] = "M",
        upsampling: bool = False,
        n_processes: int = 1,
        cleanup: bool = False,
    ) -> None:
        """ Preprocess all of the era5 POS .nc files to produce
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
        parallel: bool = True
            If true, run the preprocessing in parallel
        cleanup: bool = True
            If true, delete interim files created by the class
        """
        print(f"Reading data from {self.raw_folder}. Writing to {self.interim}")

        # get the filepaths for all of the downloaded data
        nc_files = self.get_filepaths()

        if regrid is not None:
            regrid = self.load_reference_grid(regrid)

        n_processes = max(1, n_processes)
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
