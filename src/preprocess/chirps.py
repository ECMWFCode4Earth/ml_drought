"""
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
from .preprocess_utils import select_bounding_box_xarray


class CHIRPSPreprocesser(BasePreProcessor):
    """ Preprocesses the CHIRPS data """

    def __init__(self, data_folder: Path = Path('data')) -> None:
        super().__init__(data_folder)

        self.out_dir = self.interim_folder / "chirps_preprocessed"
        if not self.out_dir.exists():
            self.out_dir.mkdir()

        self.chirps_interim = self.interim_folder / "chirps"
        if not self.out_dir.exists():
            self.out_dir.mkdir()

    def get_chirps_filepaths(self) -> List[Path]:
        return [f for f in (self.raw_folder / "chirps") .glob('*.nc')]

    @staticmethod
    def create_filename(netcdf_filepath: str,
                        subset: bool = False,
                        subset_name: Optional[str] = None):
        """
        chirps-v2.0.2009.pentads.nc
            =>
        chirps-v2.0.2009.pentads_kenya.nc
        """
        if netcdf_filepath[-3:] == '.nc':
            filename_stem = netcdf_filepath[:-3]
        else:
            filename_stem = netcdf_filepath

        if subset:
            assert subset_name is not None, "If you have set subset=True \
                then you need to assign a subset name"
            new_filename = f"{filename_stem}_{subset_name}.nc"
        else:
            new_filename = f"{filename_stem}.nc"
        return new_filename

    def preprocess_CHIRPS_data(self,
                            netcdf_filepath: Path,
                            output_dir: str,
                            subset: str = 'kenya') -> None:
        """Run the Preprocessing steps for the CHIRPS data

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

        # 2. chop out EastAfrica
        if subset == 'kenya':
            kenya_region = self.get_kenya()
            kenya_ds = select_bounding_box_xarray(new_ds, kenya_region)

        # 6. create the filepath and save to that location
        assert netcdf_filepath.name[-3:] == '.nc', f"filepath name should be a .nc file. Currently: {netcdf_filepath.name}"

        filename = create_filename(
            netcdf_filepath.name,
            subset=True,
            subset_name=subset
        )
        print(f"Saving to {output_dir}/{filename}")
        # TODO: change to pathlib.Path objects
        kenya_ds.to_netcdf(f"{output_dir}/{filename}")

        print(f"** Done for CHIRPS {netcdf_filepath.split('/')[-1]} **")

    def preprocess(self, subset: Optional[str] = 'kenya') -> None:
        """ Preprocess all of the CHIRPS .nc files to produce
        one subset file.

        Run in parallel
        """
        # get the filepaths for all of the downloaded data
        nc_files = self.get_chirps_filepaths()

        print(f"Reading data from {self.raw_folder}. \
            Writing to {self.interim_folder}")
        pool = multiprocessing.Pool(processes=100)
        outputs = pool.map(
            partial(self.preprocess_CHIRPS_data, subset=subset), nc_files
        )

        # print the outcome of the script to the user
        self.print_output(outputs)
        # save the list of errors to file
        self.save_errors(outputs)
