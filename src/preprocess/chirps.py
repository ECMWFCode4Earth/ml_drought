"""
- subset Kenya
- merge into one time (~500MB)
"""
from pathlib import Path
from functools import partial
import xarray as xr
import multiprocessing
from shutil import rmtree
from typing import List, Optional

from .base import BasePreProcessor, get_kenya
from .utils import select_bounding_box


class CHIRPSPreprocesser(BasePreProcessor):
    """ Preprocesses the CHIRPS data """

    def __init__(self, data_folder: Path = Path('data')) -> None:
        super().__init__(data_folder)

        self.out_dir = self.interim_folder / "chirps_preprocessed"
        if not self.out_dir.exists():
            self.out_dir.mkdir()

        self.chirps_interim = self.interim_folder / "chirps_interim"
        if not self.chirps_interim.exists():
            self.chirps_interim.mkdir()

    def get_chirps_filepaths(self, folder: str = 'raw') -> List[Path]:
        if folder == 'raw':
            target_folder = self.raw_folder / 'chirps'
        else:
            target_folder = self.chirps_interim
        return list(target_folder.glob('**/*.nc'))

    @staticmethod
    def create_filename(netcdf_filepath: str,
                        subset_name: Optional[str] = None) -> str:
        """
        chirps-v2.0.2009.pentads.nc
            =>
        chirps-v2.0.2009.pentads_kenya.nc
        """
        if netcdf_filepath[-3:] == '.nc':
            filename_stem = netcdf_filepath[:-3]
        else:
            filename_stem = netcdf_filepath

        if subset_name is not None:
            new_filename = f"{filename_stem}_{subset_name}.nc"
        else:
            new_filename = f"{filename_stem}.nc"
        return new_filename

    def _preprocess(self, netcdf_filepath: Path,
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
        print(f"** Starting work on {netcdf_filepath.as_posix().split('/')[-1]} **")
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
        print(f"Saving to {self.chirps_interim}/{filename}")
        ds.to_netcdf(self.chirps_interim / filename)

        print(f"** Done for CHIRPS {netcdf_filepath.name} **")

    def merge_all_timesteps(self, subset_kenya: bool = True,
                            resample_time: Optional[str] = None,
                            upsampling: bool = False) -> None:
        ds = xr.open_mfdataset(self.get_chirps_filepaths('interim'))

        if resample_time is not None:
            ds = self.resample_time(ds, resample_time, upsampling)

        out = self.out_dir / f'chirps{"_kenya" if subset_kenya else ""}.nc'
        ds.to_netcdf(out)
        print(f"\n**** {out} Created! ****\n")

    def preprocess(self, subset_kenya: bool = True,
                   regrid: Optional[Path] = None,
                   resample_time: Optional[str] = None,
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
        print(f'Reading data from {self.raw_folder}. Writing to {self.interim_folder}')

        # get the filepaths for all of the downloaded data
        nc_files = self.get_chirps_filepaths()

        if regrid is not None:
            regrid = self.load_reference_grid(regrid)

        if parallel:
            pool = multiprocessing.Pool(processes=100)
            outputs = pool.map(
                partial(self._preprocess,
                        subset_kenya=subset_kenya,
                        regrid=regrid),
                nc_files)
            print("\nOutputs (errors):\n\t", outputs)
        else:
            for file in nc_files:
                self._preprocess(file, subset_kenya, regrid,)

        # merge all of the timesteps
        self.merge_all_timesteps(subset_kenya, resample_time, upsampling)

        if cleanup:
            rmtree(self.chirps_interim)
