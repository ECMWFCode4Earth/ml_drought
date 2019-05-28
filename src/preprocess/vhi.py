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
from functools import partial

from xarray import Dataset, DataArray

from .base import (BasePreProcessor, get_kenya)

from typing import Any, List, Optional, Tuple, Union


from .utils import select_bounding_box


class VHIPreprocessor(BasePreProcessor):
    """ Preprocesses the VHI data """

    def __init__(self, data_folder: Path = Path('data')) -> None:
        super().__init__(data_folder)

        self.out_dir = self.interim_folder / 'vhi_preprocessed'
        if not self.out_dir.exists():
            self.out_dir.mkdir()

        self.vhi_interim = self.interim_folder / 'vhi'
        if not self.vhi_interim.exists():
            self.vhi_interim.mkdir()

    def get_vhi_filepaths(self) -> List[Path]:
        return [f for f in (self.raw_folder / 'vhi').glob('*/*.nc')]

    def preprocess_vhi_data(self,
                            netcdf_filepath: str,
                            output_dir: str,
                            subset_kenya: bool = True,
                            regrid: Optional[Dataset] = None) -> Path:
        """Run the Preprocessing steps for the NOAA VHI data

        Process:
        -------
        * assign time stamp
        * assign lat lon
        * create new dataset with these dimensions
        * Save the output file to new folder
        """
        print(f'** Starting work on {netcdf_filepath.split("/")[-1]} **')
        # 1. read in the dataset
        ds = xr.open_dataset(netcdf_filepath)

        # 2. extract the timestamp for that file (from the filepath)
        timestamp = self.extract_timestamp(ds, netcdf_filepath, use_filepath=True)

        # 3. extract the lat/lon vectors
        longitudes, latitudes = self.create_lat_lon_vectors(ds)

        # 4. create new dataset with these dimensions
        new_ds = self.create_new_dataset(ds, longitudes, latitudes, timestamp)

        # 5. chop out EastAfrica - TODO: have a dictionary of legitimate args
        if subset_kenya:
            kenya_region = get_kenya()
            new_ds = select_bounding_box(new_ds, kenya_region)

        if regrid is not None:
            new_ds = self.regrid(new_ds, regrid)

        # 6. create the filepath and save to that location
        filename = self.create_filename(
            timestamp,
            netcdf_filepath,
            subset_name='kenya' if subset_kenya else None
        )
        print(f'Saving to {output_dir}/{filename}')
        # TODO: change to pathlib.Path objects
        new_ds.to_netcdf(f'{output_dir}/{filename}')

        print(f'** Done for VHI {netcdf_filepath.split("/")[-1]} **')

        return Path(f'{output_dir}/{filename}')

    def _process(self,
                 netcdf_filepath: str,
                 subset_kenya: bool = True,
                 regrid: Optional[Dataset] = None
                 ) -> Union[Path, Tuple[Exception, str]]:
        """ function to be run in parallel & safely catch errors

        https://stackoverflow.com/a/24683990/9940782
        """
        print(f"Starting work on {netcdf_filepath}")
        if not self.vhi_interim.exists():
            self.vhi_interim.mkdir()

        if isinstance(netcdf_filepath, pathlib.PosixPath):
            netcdf_filepath = netcdf_filepath.as_posix()

        try:
            return self.preprocess_vhi_data(
                netcdf_filepath, self.vhi_interim.as_posix(), subset_kenya, regrid
            )
        except Exception as e:
            print(f"### FAILED: {netcdf_filepath}")
            return e, netcdf_filepath

    @staticmethod
    def print_output(outputs: List) -> None:
        print("\n\n*************************\n\n")
        print("Script Run")
        print("*************************")
        print("Paths:")
        print("\nPaths: ", [o for o in outputs if o is not None])
        print("\n__Failed File List:",
              [o[-1] for o in outputs if not isinstance(o, Path)])

    def save_errors(self, outputs: List) -> Path:
        # write output of failed files to python.txt
        with open(self.interim_folder / 'vhi_preprocess_errors.pkl', 'wb') as f:
            pickle.dump([error[-1] for error in outputs if error is not None], f)

        return self.interim_folder / 'vhi_preprocess_errors.pkl'

    def merge_to_one_file(self, region: Optional[str] = None) -> Dataset:
        # TODO how do we figure out the misisng timestamps?
        # 1) find the anomalous gaps in the timesteps (> 7 days)
        # 2) find the years where there are less than 52 timesteps
        nc_files = [f for f in self.vhi_interim.glob('*')]
        nc_files.sort()
        ds = xr.open_mfdataset(nc_files)

        if region is not None:
            outpath = self.out_dir / f"vhi_preprocess.nc"
        else:
            outpath = self.out_dir / f"vhi_preprocess_{region}.nc"

        # save the merged filepath
        ds.to_netcdf(outpath)
        print(f"Timesteps merged and saved: {outpath}")

        # turn from a dask mfDataset to a Dataset (how is this done?)

        return ds

    def preprocess(self,
                   subset_kenya: bool = True,
                   regrid: Optional[Dataset] = None) -> None:
        """ Preprocess all of the NOAA VHI .nc files to produce
        one subset file with consistent lat/lon and timestamps.

        Run in parallel

        Arguments
        ----------
        subset_kenya: bool = True
            Whether to subset Kenya when preprocessing
        regrid: Optional[Dataset] = None
            If a dataset is passed, the VHI files will be regridded to have the same
            grid as that dataset. If None, no regridding happens
        """
        # get the filepaths for all of the downloaded data
        nc_files = self.get_vhi_filepaths()

        print(f"Reading data from {self.raw_folder}. \
            Writing to {self.interim_folder}")
        pool = multiprocessing.Pool(processes=100)
        outputs = pool.map(partial(self._process,
                                   subset_kenya=subset_kenya,
                                   regrid=regrid), nc_files)
        errors = [o for o in outputs if not isinstance(o, Path)]

        # TODO check how these errors are being saved (now all paths returned)
        # print the outcome of the script to the user
        self.print_output(outputs)
        # save the list of errors to file
        self.save_errors(errors)

    @staticmethod
    def create_filename(t: Timestamp,
                        netcdf_filepath: str,
                        subset_name: Optional[str] = None):
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
        substr = netcdf_filepath.split('/')[-1].split('.P')[0]
        if subset_name is not None:
            new_filename = f"STAR_{substr}_{t.year}_{t.month}_{t.day}_{subset_name}_VH.nc"
        else:
            new_filename = f"STAR_{substr}_{t.year}_{t.month}_{t.day}_VH.nc"
        return new_filename

    @staticmethod
    def extract_timestamp(ds: Dataset,
                          netcdf_filepath: str,
                          use_filepath: bool = True,
                          time_begin: bool = True) -> Timestamp:
        """from the `attrs` or filename create a datetime object for acquisition time.

        NOTE: the acquisition date is SOMEWHERE in this time range (satuday-friday)

        USE THE FILENAME
        """

        if use_filepath:  # use the weeknumber in filename
            # https://stackoverflow.com/a/22789330/9940782
            YYYYWWW = netcdf_filepath.split('P')[-1].split('.')[0]
            year = YYYYWWW[:4]
            week = YYYYWWW[5:7]
            atime = time.asctime(
                time.strptime('{} {} 1'.format(year, week), '%Y %W %w')
            )

        else:
            year = ds.attrs['YEAR']
            if time_begin:
                day_num = ds.attrs['DATE_BEGIN']
            else:  # time_end
                day_num = ds.attrs['DATE_END']

            atime = time.asctime(
                time.strptime('{} {}'.format(year, day_num), '%Y %j')
            )

        date = pd.to_datetime(atime)
        return date

    @staticmethod
    def create_lat_lon_vectors(ds: Dataset) -> Tuple[Any, Any]:
        """ read the `ds.attrs` and create new latitude, longitude vectors """
        assert ds.WIDTH.size == 10000, \
            f'We are hardcoding the lat/lon values so we need to ensure that all dims ' \
            f'are the same. WIDTH != 10000, == {ds.WIDTH.size}'
        assert ds.HEIGHT.size == 3616, \
            f'We are hardcoding the lat/lon values so we need to ensure that all dims ' \
            f'are the same. HEIGHT != 3616, == {ds.HEIGHT.size}'

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
    def create_new_dataarray(ds: Dataset,
                             variable: str,
                             longitudes: np.ndarray,
                             latitudes: np.ndarray,
                             timestamp: Timestamp) -> DataArray:
        """ Create a new dataarray for the `variable` from `ds` with geocoding and timestamp """
        # Assert statements - to a test function?
        dims = list(ds.dims)
        variables = list(ds.variables.keys())

        assert variable in variables, \
            f'variable: {variable} need to be a variable in the ds! Currently {variables}'
        assert (ds[dims[0]].size == longitudes.size) or (ds[dims[1]].size == longitudes.size), \
            f'Size of dimensions {dims} should be equal either to the size of longitudes.' \
            f' \n Currently longitude: {longitudes.size}. {ds[dims[0]]}: {ds[dims[0]].size},' \
            f' {ds[dims[1]]}: {ds[dims[1]].size}'
        assert (ds[dims[0]].size == latitudes.size) or (ds[dims[1]].size == latitudes.size), \
            f'Size of dimensions {dims} should be equal either to the size of latitudes' \
            f'. \n Currently latitude: {latitudes.size}. {ds[dims[0]]}: {ds[dims[0]].size},' \
            f' {ds[dims[1]]}: {ds[dims[1]].size}'
        assert np.array(timestamp).size == 1, \
            'The function only currently works with SINGLE TIMESTEPS.'

        da = xr.DataArray(
            [ds[variable].values],
            dims=['time', 'lat', 'lon'],
            coords={'lon': longitudes,
                    'lat': latitudes,
                    'time': [timestamp]}
        )
        da.name = variable
        return da

    def create_new_dataset(self,
                           ds: Dataset,
                           longitudes: np.ndarray,
                           latitudes: np.ndarray,
                           timestamp: Timestamp,
                           all_vars: bool = False) -> Dataset:
        """ Create a new dataset from ALL the variables in `ds` with the dims"""
        # initialise the list
        da_list = []

        # for each variable create a new data array and append to list
        if all_vars:
            variables = list(ds.variables.keys())
        else:
            variables = ['VHI']
        for variable in variables:
            da_list.append(self.create_new_dataarray(ds, variable, longitudes,
                                                     latitudes, timestamp))
        # merge all of the variables into one dataset
        new_ds = xr.merge(da_list)
        new_ds.attrs = ds.attrs

        return new_ds
