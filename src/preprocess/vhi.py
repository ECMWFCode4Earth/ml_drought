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
from typing import List, Optional
import pickle
from functools import partial

from xarray import Dataset

from .base import (BasePreProcessor,)
from .preprocess_vhi import (
    extract_timestamp,
    create_lat_lon_vectors,
    create_new_dataset,
    create_filename,
)
from .preprocess_utils import select_bounding_box_xarray


class VHIPreprocessor(BasePreProcessor):
    """ Preprocesses the VHI data """

    def __init__(self, data_folder: Path = Path('data')) -> None:
        super().__init__(data_folder)

        self.out_dir = self.interim_folder / "vhi_preprocessed"
        if not self.out_dir.exists():
            self.out_dir.mkdir()

        self.vhi_interim = self.interim_folder / "vhi"
        if not self.vhi_interim.exists():
            self.vhi_interim.mkdir()

    def get_vhi_filepaths(self) -> List[Path]:
        return [f for f in (self.raw_folder / "vhi") .glob('*/*.nc')]

    def preprocess_VHI_data(self,
                            netcdf_filepath: str,
                            output_dir: str,
                            subset: str = 'kenya') -> None:
        """Run the Preprocessing steps for the NOAA VHI data

        Process:
        -------
        * assign time stamp
        * assign lat lon
        * create new dataset with these dimensions
        * Save the output file to new folder
        """
        print(f"** Starting work on {netcdf_filepath.split('/')[-1]} **")
        # 1. read in the dataset
        ds = xr.open_dataset(netcdf_filepath)

        # 2. extract the timestamp for that file (from the filepath)
        timestamp = extract_timestamp(ds, netcdf_filepath, use_filepath=True)

        # 3. extract the lat/lon vectors
        longitudes, latitudes = create_lat_lon_vectors(ds)

        # 4. create new dataset with these dimensions
        new_ds = create_new_dataset(ds, longitudes, latitudes, timestamp)

        # 5. chop out EastAfrica - TODO: have a dictionary of legitimate args
        if subset == 'kenya':
            kenya_region = self.get_kenya()
            kenya_ds = select_bounding_box_xarray(new_ds, kenya_region)

        # 6. create the filepath and save to that location
        filename = create_filename(
            timestamp,
            netcdf_filepath,
            subset=True,
            subset_name=subset
        )
        print(f"Saving to {output_dir}/{filename}")
        # TODO: change to pathlib.Path objects
        kenya_ds.to_netcdf(f"{output_dir}/{filename}")

        print(f"** Done for VHI {netcdf_filepath.split('/')[-1]} **")

    def add_coordinates(self, netcdf_filepath: str, subset: str = 'kenya'):
        """ function to be run in parallel & safely catch errors

        https://stackoverflow.com/a/24683990/9940782
        """
        print(f"Starting work on {netcdf_filepath}")
        vhi_interim_folder = self.interim_folder / "vhi"
        if not vhi_interim_folder.exists():
            vhi_interim_folder.mkdir()

        if isinstance(netcdf_filepath, pathlib.PosixPath):
            netcdf_filepath = netcdf_filepath.as_posix()

        try:
            return self.preprocess_VHI_data(
                netcdf_filepath, vhi_interim_folder.as_posix(),
            )
        except Exception as e:
            print(f"### FAILED: {netcdf_filepath}")
            return e, netcdf_filepath

    @staticmethod
    def print_output(outputs: List) -> None:
        print("\n\n*************************\n\n")
        print("Script Run")
        print("*************************")
        print("Errors:")
        print("\nError: ", [error for error in outputs if error is not None])
        print("\n__Failed File List:",
              [error[-1] for error in outputs if error is not None])

    def save_errors(self, outputs: List) -> Path:
        # write output of failed files to python.txt
        with open(self.interim_folder / 'vhi_preprocess_errors.pkl', 'wb') as f:
            pickle.dump([error[-1] for error in outputs if error is not None], f)

        return self.interim_folder / 'vhi_preprocess_errors.pkl'

    def merge_to_one_file(self,
                          region: Optional[str] = None) -> Dataset:
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

    def preprocess(self, subset: Optional[str] = 'kenya') -> None:
        """ Preprocess all of the NOAA VHI .nc files to produce
        one subset file with consistent lat/lon and timestamps.

        Run in parallel
        """
        # get the filepaths for all of the downloaded data
        nc_files = self.get_vhi_filepaths()

        print(f"Reading data from {self.raw_folder}. \
            Writing to {self.interim_folder}")
        pool = multiprocessing.Pool(processes=100)
        outputs = pool.map(
            partial(self.add_coordinates, subset=subset), nc_files
        )

        # print the outcome of the script to the user
        self.print_output(outputs)
        # save the list of errors to file
        self.save_errors(outputs)
