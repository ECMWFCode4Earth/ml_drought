"""
- subset Kenya
- merge into one time (~500MB)
"""
from pathlib import Path
import xarray as xr
import multiprocessing
from typing import List, Optional
import re

from .base import BasePreProcessor, get_kenya
from .utils import select_bounding_box


class CHIRPSPreprocesser(BasePreProcessor):
    """ Preprocesses the CHIRPS data """

    def __init__(self, data_folder: Path = Path('data')) -> None:
        super().__init__(data_folder)

        self.out_dir = self.interim_folder / "chirps_preprocessed"
        if not self.out_dir.exists():
            self.out_dir.mkdir()

        self.chirps_interim = self.interim_folder / "chirps"
        if not self.chirps_interim.exists():
            self.chirps_interim.mkdir()

    def get_chirps_filepaths(self) -> List[Path]:
        return list((self.raw_folder / "chirps") .glob('**/*.nc'))

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
                    regrid: Optional[xr.Dataset] = None,
                    resample_time: Optional[str] = None,
                    upsampling: bool = False) -> None:
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

        if resample_time is not None:
            ds = self.resample_time(ds, resample_time, upsampling)

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

    @staticmethod
    def get_year_from_filename(filename: str) -> int:
        years = re.compile(r'\d{4}')
        year = int(years.findall(filename)[0])

        assert (1981 <= year <= 2020), \
            f"year should be between 1981-2020 (CHIRPS does not extend to before 1981). " \
            f"Currently: {year}"

        return year

    def merge_all_timesteps(self, min_year: int, max_year: int,
                            subset_kenya: bool = True) -> None:
        ds = xr.open_mfdataset(
            [f for f in self.chirps_interim.glob('*.nc')]
        )
        print(f"Merging timesteps from {min_year} to {max_year}")

        out = self.out_dir / f'chirps_{min_year}{max_year}{"_kenya" if subset_kenya else ""}.nc'
        ds.to_netcdf(out)
        print(f"\n**** {out} Created! ****\n")

    def preprocess(self, subset_kenya: bool = True,
                   regrid: Optional[xr.Dataset] = None,
                   resample_time: Optional[str] = None,
                   upsampling: bool = False,
                   parallel: bool = False) -> None:
        """ Preprocess all of the CHIRPS .nc files to produce
        one subset file.

        Run in parallel
        """
        print(f'Reading data from {self.raw_folder}. Writing to {self.interim_folder}')

        # get the filepaths for all of the downloaded data
        nc_files = self.get_chirps_filepaths()

        # preprocess chirps files (subset region) in parallel
        if parallel:
            # TODO: pass the arguments using iterators
            pool = multiprocessing.Pool(processes=100)
            outputs = pool.map(
                self._preprocess, nc_files
            )
            print("\nOutputs (errors):\n\t", outputs)
        else:
            for file in nc_files:
                self._preprocess(file, subset_kenya, regrid, resample_time, upsampling)

        # merge all of the timesteps
        if len(nc_files) > 1:
            years = [self.get_year_from_filename(f.name) for f in nc_files]
            min_year = min(years)
            max_year = max(years)
            self.merge_all_timesteps(min_year, max_year, subset_kenya)
