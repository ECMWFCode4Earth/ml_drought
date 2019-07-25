from pathlib import Path
import xarray as xr
import multiprocessing
from functools import partial
from typing import Optional
from shutil import rmtree

from .base import BasePreProcessor


class ERA5LandPreprocessor(BasePreProcessor):
    """ Preprocesses the ERA5 Land data """
    dataset = 'reanalysis-era5-land'

    @staticmethod
    def create_filename(netcdf_filepath: Path,
                        subset_name: Optional[str] = None) -> str:

        var_name = netcdf_filepath.parts[-3]
        months = netcdf_filepath.parts[-1][:-3]
        year = netcdf_filepath.parts[-2]

        stem = f'{year}_{months}_{var_name}'
        if subset_name is not None:
            stem = f'{stem}_{subset_name}'
        return f'{stem}.nc'

    def _preprocess_single(self, netcdf_filepath: Path,
                           subset_str: Optional[str] = 'kenya',
                           regrid: Optional[xr.Dataset] = None) -> None:
        """ Preprocess a single netcdf file (run in parallel if
        `parallel_processes` arg > 1)

        Process:
        -------
        * rename latitude/longitude -> lat/lon
        * chop region of interset (ROI)
        * regrid to same spatial grid as a reference dataset (`regrid`)
        * Save the output file to new folder / filename

        Todo:
        # read the variable name from the fpath
        # variable = netcdf_filepath.parents[1].name
        """
        print(f'Processing {netcdf_filepath.name}')

        # 1. read in the dataset
        ds = xr.open_dataset(netcdf_filepath).rename({'longitude': 'lon', 'latitude': 'lat'})

        # 2. chop out EastAfrica
        if subset_str is not None:
            ds = self.chop_roi(ds, subset_str, inverse_lat=True)

        if regrid is not None:
            ds = self.regrid(ds, regrid)

        filename = self.create_filename(
            netcdf_filepath,
            subset_name=subset_str if subset_str is not None else None
        )
        print(f'Saving to {self.interim}/{filename}')
        ds.to_netcdf(self.interim / filename)

        print(f'Done for ERA5-Land {netcdf_filepath.name}')

    def preprocess(self, subset_str: Optional[str] = 'kenya',
                   regrid: Optional[Path] = None,
                   resample_time: Optional[str] = 'M',
                   upsampling: bool = False,
                   parallel_processes: int = 1,
                   cleanup: bool = True) -> None:
        """Preprocess all of the ERA5-Land .nc files to produce
        one subset file.

        Note:
        ----
        - the raw data is downloaded at annual resolution by default
        """
        print(f'Reading data from {self.raw_folder}. Writing to {self.interim}')
        nc_files = self.get_filepaths()

        if regrid is not None:
            regrid = self.load_reference_grid(regrid)

        # parallel processing ?
        if parallel_processes <= 1:  # sequential
            for file in nc_files:
                self._preprocess_single(file, subset_str, regrid)
        else:
            pool = multiprocessing.Pool(processes=parallel_processes)
            outputs = pool.map(
                partial(self._preprocess_single,
                        subset_str=subset_str,
                        regrid=regrid),
                nc_files)
            print("\nOutputs (errors):\n\t", outputs)

        # merge and resample files
        self.merge_files(
            subset_str=subset_str,
            resample_time=resample_time,
            upsampling=upsampling
        )

        if cleanup:
            rmtree(self.interim)
