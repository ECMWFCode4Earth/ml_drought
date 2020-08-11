"""

- Add lat lon coordinates
- add time coordinates
- subset Kenya
- merge into one time (~500MB)
"""
from pathlib import Path
import pathlib
import xarray as xr
import multiprocessing
import numpy as np
import pickle
import pandas as pd
from pandas._libs.tslibs.timestamps import Timestamp
import time
from shutil import rmtree
from functools import partial

from xarray import Dataset, DataArray

from .base import BasePreProcessor

from typing import Any, List, Optional, Tuple, Union


class VHIPreprocessor(BasePreProcessor):
    """ Preprocesses the VHI data """

    dataset = "vhi"

    raw_height: int = 3616
    raw_width: int = 10000

    def __init__(self, data_folder: Path = Path("data"), var: str = "VHI") -> None:
        """
        var: str
            The variable to output. This will be output to its own
            folder, so it is safe to run this preprocessor for each var
            (the data won't be overwritten). Must be one of {'VCI', 'VHI', 'TCI'}
        """
        assert var in ["VCI", "VHI", "TCI"]
        self.data_var = var

        super().__init__(data_folder, var)

    def _preprocess(
        self,
        netcdf_filepath: str,
        output_dir: str,
        subset_str: Optional[str] = "kenya",
        regrid: Optional[Dataset] = None,
    ) -> Path:
        """Run the Preprocessing steps for the NOAA VHI data

        Process:
        -------
        * assign time stamp
        * assign lat lon
        * create new dataset with these dimensions
        * Save the output file to new folder
        """
        print(f'Starting work on {netcdf_filepath.split("/")[-1]}')
        # 1. read in the dataset
        ds = xr.open_dataset(netcdf_filepath)
        # FLIP the `HEIGHT` array
        ds = ds.sortby("HEIGHT", ascending=False)

        # 2. extract the timestamp for that file (from the filepath)
        timestamp = self.extract_timestamp(ds, netcdf_filepath, use_filepath=True)

        # 3. create the filepath
        filename = self.create_filename(
            timestamp, netcdf_filepath, subset_name=subset_str
        )

        # test if the file already exists
        if Path(f"{output_dir}/{filename}").exists():
            print(f"{output_dir}/{filename} Already exists!")
            return Path(f"{output_dir}/{filename}")

        # 4. extract the lat/lon vectors
        longitudes, latitudes = self.create_lat_lon_vectors(ds)

        # 5. create new dataset with these dimensions
        new_ds = self.create_new_dataset(
            ds, longitudes, latitudes, timestamp, [self.data_var]
        )

        # 6. chop out EastAfrica
        if subset_str is not None:
            new_ds = self.chop_roi(new_ds, subset_str)

        if regrid is not None:
            new_ds = self.regrid(new_ds, regrid)

        # 7. save to filepath location
        print(f"Saving to {output_dir}/{filename}")
        # TODO: change to pathlib.Path objects
        new_ds.to_netcdf(f"{output_dir}/{filename}")

        print(f'** Done for VHI {netcdf_filepath.split("/")[-1]} **')

        return Path(f"{output_dir}/{filename}")

    def _preprocess_wrapper(
        self,
        netcdf_filepath: str,
        subset_str: Optional[str] = "kenya",
        regrid: Optional[Dataset] = None,
    ) -> Union[Path, Tuple[Exception, str]]:
        """ function to be run in parallel & safely catch errors

        https://stackoverflow.com/a/24683990/9940782
        """
        print(f"Starting work on {netcdf_filepath}")
        if not self.interim.exists():
            self.interim.mkdir()

        if isinstance(netcdf_filepath, pathlib.PosixPath):
            netcdf_filepath = netcdf_filepath.as_posix()

        try:
            return self._preprocess(
                netcdf_filepath, self.interim.as_posix(), subset_str, regrid
            )
        except Exception as e:
            print(f"### FAILED: {netcdf_filepath}")
            return e, netcdf_filepath

    def preprocess(
        self,
        subset_str: Optional[str] = "kenya",
        regrid: Optional[Path] = None,
        n_parallel_processes: int = 1,
        resample_time: Optional[str] = "M",
        upsampling: bool = False,
        cleanup: bool = True,
    ) -> None:
        """ Preprocess all of the NOAA VHI .nc files to produce
        one subset file with consistent lat/lon and timestamps.

        Run in parallel if n_parallel_processes > 1

        Arguments
        ----------
        subset_str: Optional[str] = 'kenya'
            Region to subset. Currently valid: {'kenya', 'ethiopia', 'east_africa'}
        regrid: Optional[Path] = None
            If a Path is passed, the VHI files will be regridded to have the same
            grid as the dataset at that Path. If None, no regridding happens
        resample_time: str = 'M'
            If not None, defines the time length to which the data will be resampled
        upsampling: bool = False
            If true, tells the class the time-sampling will be upsampling. In this case,
            nearest instead of mean is used for the resampling
        n_parallel_processes: int = 1
            If > 1, run the preprocessing in n_parallel_processes
        cleanup: bool = True
            If true, delete interim files created by the class
        """
        # get the filepaths for all of the downloaded data
        nc_files = self.get_filepaths()

        if regrid is not None:
            regrid = self.load_reference_grid(regrid)

        print(
            f"Reading data from {self.raw_folder}. \
            Writing to {self.interim}"
        )
        n_parallel_processes = max(n_parallel_processes, 1)
        if n_parallel_processes > 1:
            pool = multiprocessing.Pool(processes=n_parallel_processes)
            outputs = pool.map(
                partial(self._preprocess_wrapper, subset_str=subset_str, regrid=regrid),
                nc_files,
            )
            errors = [o for o in outputs if not isinstance(o, Path)]

            # TODO check how these errors are being saved (now all paths returned)
            # print the outcome of the script to the user
            self.print_output(outputs)
            # save the list of errors to file
            self.save_errors(errors)
        else:
            for file in nc_files:
                output_dir = self.interim.as_posix()
                try:
                    self._preprocess(
                        str(file),
                        subset_str=subset_str,
                        regrid=regrid,
                        output_dir=output_dir,
                    )
                except OSError as e:
                    print(e)
                    print(f"{e} Error for {file}. Skipping")

        self.merge_files(
            subset_str=subset_str, resample_time=resample_time, upsampling=upsampling
        )
        if cleanup:
            rmtree(self.interim)

    @staticmethod
    def print_output(outputs: List) -> None:
        print("\n\n*************************\n\n")
        print("Script Run")
        print("*************************")
        print("Paths:")
        print("\nPaths: ", [o for o in outputs if o is not None])
        print(
            "\n__Failed File List:", [o[-1] for o in outputs if not isinstance(o, Path)]
        )

    def save_errors(self, outputs: List) -> Path:
        # write output of failed files to python.txt
        with open(self.interim / "vhi_preprocess_errors.pkl", "wb") as f:
            pickle.dump([error[-1] for error in outputs if error is not None], f)

        return self.interim / "vhi_preprocess_errors.pkl"

    @staticmethod
    def create_filename(
        t: Timestamp, netcdf_filepath: str, subset_name: Optional[str] = None
    ) -> str:
        """ create a sensible output filename (HARDCODED for this problem)
        Arguments:
        ---------
        t : pandas._libs.tslibs.timestamps.Timestamp, datetime.datetime
            timestamp of this netcdf file

        Example Output:
        --------------
        STAR_VHP.G04.C07.NN.P_20110101_VH.nc
        VHP.G04.C07.NJ.P1996027.VH.nc
        """
        substr = netcdf_filepath.split("/")[-1].split(".P")[0]
        if subset_name is not None:
            new_filename = (
                f"STAR_{substr}_{t.year}_{t.month}_{t.day}_{subset_name}_VH.nc"
            )
        else:
            new_filename = f"STAR_{substr}_{t.year}_{t.month}_{t.day}_VH.nc"
        return new_filename

    @staticmethod
    def extract_timestamp(
        ds: Dataset,
        netcdf_filepath: str,
        use_filepath: bool = True,
        time_begin: bool = True,
    ) -> Timestamp:
        """from the `attrs` or filename create a datetime object for acquisition time.

        NOTE: the acquisition date is SOMEWHERE in this time range (satuday-friday)

        USE THE FILENAME
        """

        if use_filepath:  # use the weeknumber in filename
            # https://stackoverflow.com/a/22789330/9940782
            YYYYWWW = netcdf_filepath.split("P")[-1].split(".")[0]
            year = YYYYWWW[:4]
            week = YYYYWWW[5:7]
            atime = time.asctime(
                time.strptime("{} {} 1".format(year, week), "%Y %W %w")
            )

        else:
            year = ds.attrs["YEAR"]
            if time_begin:
                day_num = ds.attrs["DATE_BEGIN"]
            else:  # time_end
                day_num = ds.attrs["DATE_END"]

            atime = time.asctime(time.strptime("{} {}".format(year, day_num), "%Y %j"))

        date = pd.to_datetime(atime)
        return date

    def create_lat_lon_vectors(self, ds: Dataset) -> Tuple[Any, Any]:
        """ read the `ds.attrs` and create new latitude, longitude vectors """
        assert ds.WIDTH.size == self.raw_width, (
            f"We are hardcoding the lat/lon values so we need to ensure that all dims "
            f"are the same. WIDTH != 10000, == {ds.WIDTH.size}"
        )
        assert ds.HEIGHT.size == self.raw_height, (
            f"We are hardcoding the lat/lon values so we need to ensure that all dims "
            f"are the same. HEIGHT != 3616, == {ds.HEIGHT.size}"
        )

        # NOTE: hardcoded for the VHI data (some files don't have the attrs)
        lonmin, lonmax = -180.0, 180.0
        latmin, latmax = -55.152, 75.024

        # extract the size of the lat/lon coords
        lat_len, lon_len = ds.HEIGHT.shape[0], ds.WIDTH.shape[0]

        # create the vector
        longitudes = np.linspace(lonmin, lonmax, lon_len)
        latitudes = np.linspace(latmin, latmax, lat_len)

        return longitudes, latitudes

    @staticmethod
    def create_new_dataarray(
        ds: Dataset,
        variable: str,
        longitudes: np.ndarray,
        latitudes: np.ndarray,
        timestamp: Timestamp,
    ) -> DataArray:
        """ Create a new dataarray for the `variable` from `ds` with geocoding and timestamp """
        # Assert statements - to a test function?
        dims = list(ds.dims)
        variables = list(ds.variables.keys())

        assert (
            variable in variables
        ), f"variable: {variable} need to be a variable in the ds! Currently {variables}"
        assert (ds[dims[0]].size == longitudes.size) or (
            ds[dims[1]].size == longitudes.size
        ), (
            f"Size of dimensions {dims} should be equal either to the size of longitudes."
            f" \n Currently longitude: {longitudes.size}. {ds[dims[0]]}: {ds[dims[0]].size},"
            f" {ds[dims[1]]}: {ds[dims[1]].size}"
        )
        assert (ds[dims[0]].size == latitudes.size) or (
            ds[dims[1]].size == latitudes.size
        ), (
            f"Size of dimensions {dims} should be equal either to the size of latitudes"
            f". \n Currently latitude: {latitudes.size}. {ds[dims[0]]}: {ds[dims[0]].size},"
            f" {ds[dims[1]]}: {ds[dims[1]].size}"
        )
        assert (
            np.array(timestamp).size == 1
        ), "The function only currently works with SINGLE TIMESTEPS."

        da = xr.DataArray(
            [ds[variable].values],
            dims=["time", "lat", "lon"],
            coords={"lon": longitudes, "lat": latitudes, "time": [timestamp]},
        )
        da.name = variable
        return da

    def create_new_dataset(
        self,
        ds: Dataset,
        longitudes: np.ndarray,
        latitudes: np.ndarray,
        timestamp: Timestamp,
        var_selection: Optional[List[str]] = None,
    ) -> Dataset:
        """ Create a new dataset from ALL the variables in `ds` with the dims.
            If no vars are selected, all are used"""
        # initialise the list
        da_list = []

        # for each variable create a new data array and append to list
        if var_selection is None:
            var_selection = list(ds.variables.keys())
        for variable in var_selection:
            da_list.append(
                self.create_new_dataarray(
                    ds, variable, longitudes, latitudes, timestamp
                )
            )
        # merge all of the variables into one dataset
        new_ds = xr.merge(da_list)
        new_ds.attrs = ds.attrs

        return new_ds
