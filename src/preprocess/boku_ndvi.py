"""
Storing the image values in images as real values generally results
in large files given the space needs for storing floats. The NDVI
netCDF files therefore have their values scaled to have integers.
You need to know the flags for missing data [255], inland water [252] and
Ocean[251] and then scale the integers to get the actual NDVI values.

The equation for the scaling is as show below.
𝑽𝑰=𝑽𝑰𝒔𝒍𝒐𝒑𝒆∙𝑫𝑵+ 𝑽𝑰𝒊𝒏𝒕𝒆𝒓𝒄𝒆𝒑𝒕 where DN is the digital number on the image.

The actual formula would look like below:
NDVI = 0.0048 * DN- 0.200
"""

from pathlib import Path
from shutil import rmtree
import xarray as xr
import multiprocessing
from functools import partial
import re
from datetime import datetime
import pandas as pd
import numpy as np

from typing import Optional

from .base import BasePreProcessor
from .dekad_utils import dekad_index

from src.analysis import ConditionIndex
from typing import Union, Tuple


class BokuNDVIPreprocessor(BasePreProcessor):
    def __init__(
        self,
        data_folder: Path = Path("data"),
        output_name: Optional[str] = None,
        resolution: str = "1000",
        downsample_first: bool = False,
    ):
        self.resolution = str(resolution)
        self.static = False
        self.downsample_first = downsample_first

        if self.resolution == "1000":
            # 1km pixel
            self.dataset: str = "boku_ndvi_1000"  # type: ignore
        elif self.resolution == "250":
            # 250m pixel
            self.dataset: str = "boku_ndvi_250"  # type: ignore
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
        extract the datetime from filename (https://strftime.org/)

        Example:
        1000m
            MCD13A2.t200915.006.EAv1.1_km_10_days_NDVI.O1.nc
            the 15th Monday of 2009 (NO)
            the 15th Dekad of 2009
            https://pytesmo.readthedocs.io/en/latest/_modules/pytesmo/timedate/dekad.html
        250m
            MCD09Q1.A2010319.006.KEHOA.250m_07_days_NDVI.Bw_TMP.nc
            day 319 of 2010

        returns datetime object
        """

        if self.resolution == "1000":
            # GET the Dekad
            # regex pattern (4 digits after '.t')
            year_pattern = re.compile(r".t\d{4}")
            # extract the year from the filename
            year = year_pattern.findall(filename)[0].split(".t")[-1]
            # create a list of DEKAD datetimes
            begin = pd.to_datetime(f"{year}-01-01").to_pydatetime()
            end = pd.to_datetime(f"{int(year) + 1}-01-01").to_pydatetime()
            dekad_list = dekad_index(begin, end)
            # extract the dekad_number
            dekad_num = year_pattern.split(filename)[-1].split(".")[0]

            # index the list of dates by the dekad_number
            return dekad_list[int(dekad_num)]

        elif self.resolution == "250":
            # regex pattern (4 digits after '.t')
            year_pattern = re.compile(r".A\d{4}")
            # extract the year from the filename
            year = year_pattern.findall(filename)[0].split(".A")[-1]

            # extract the day_number
            day_num = year_pattern.split(filename)[-1].split(".")[0]

            return datetime.strptime(f"{year}-{day_num}", "%Y-%j")

        else:
            assert False, "Only working with two resolutions 1000 / 250"

    def create_new_dataarray(
        self, ds: xr.Dataset, timestamp: pd.Timestamp
    ) -> xr.Dataset:
        variable = "boku_ndvi"
        valid_vars = [v for v in ds.data_vars]
        if variable not in valid_vars:
            variable = "modis_ndvi"
        assert (
            variable in valid_vars
        ), "Expect modis_ndvi / boku_ndvi to be the variable"

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

    def _convert_to_VCI(
        self, ds: xr.Dataset, rolling_window: int = 1, variable: Optional[str] = None
    ) -> xr.Dataset:
        """Convert the BOKU NDVI data to VCI data
        """
        vci = ConditionIndex(ds=ds, resample_str=None)
        if variable is None:
            variable = [v for v in ds.data_vars][0]
        vci.fit(variable=variable, rolling_window=rolling_window)
        var_ = [v for v in vci.index.data_vars][0]
        vci = vci.index.rename({var_: f"VCI"})

        return vci

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

        # 4. mask out missing values
        # 251=ocean, 252=inland_water, 255=missing_data
        ds = ds.where((ds != 251) & (ds != 252) & (ds != 255))

        # 5. convert from int to ndvi float (storage reasons)
        # 𝑽𝑰 = 𝑽𝑰𝒔𝒍𝒐𝒑𝒆 * value + 𝑽𝑰𝒊𝒏𝒕𝒆𝒓𝒄𝒆𝒑𝒕
        ds = (0.0048 * ds) - 0.200

        # 7. create the filepath and save to that location
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

    def merge_files(
        self,
        subset_str: Optional[str] = "kenya",
        resample_time: Optional[str] = "M",
        upsampling: bool = False,
        filename: Optional[str] = None,
    ) -> Union[Path, Tuple[Path]]:
        """Unique merge files because want to calculate the VCI BEFORE
        we decrease the temporal resolution
        """

        ds = xr.open_mfdataset(
            self.get_filepaths("interim"), combine="nested", concat_dim="time"
        )

        if not self.downsample_first:
            # vci1,
            vci = self._convert_to_VCI(ds).rename({f"VCI": "boku_VCI"})
            assert vci.isnull().mean() < 1, "All NaN values!"
            ds = xr.auto_combine([ds, vci])
            # vci3m
            vci = self._convert_to_VCI(
                ds, rolling_window=3, variable="boku_VCI"
            ).rename({f"VCI": "VCI3M"})
            assert vci.isnull().mean() < 1, "All NaN values!"
            ds = xr.auto_combine([ds, vci])

        if resample_time is not None:
            ds = self.resample_time(ds, resample_time, upsampling)

        if filename is None:
            filename = f'data{"_" + subset_str if subset_str is not None else ""}.nc'
        out = self.out_dir / filename

        ds.to_netcdf(out)
        print(f"\n**** {out} Created! ****\n")

        return out

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
        nc_files.sort()

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
            for ix, file in enumerate(nc_files):
                print(f"INDEX: {ix}")
                self._preprocess_single(file, subset_str, regrid)

        # merge all of the timesteps
        outpath = self.merge_files(subset_str, resample_time, upsampling)

        # 6. add in the VCI data too
        if self.downsample_first:
            # downsample BEFORE calculating VCI
            ds = xr.open_dataset(outpath)
            vci = self._convert_to_VCI(ds).rename({f"VCI": "boku_VCI"})
            assert vci.isnull().mean() < 1, "All NaN values!"
            ds = xr.auto_combine([ds, vci])

        if cleanup:
            rmtree(self.interim)
