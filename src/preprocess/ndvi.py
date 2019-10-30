from pathlib import Path
import xarray as xr
import numpy as np
import pandas as pd

import multiprocessing
from functools import partial
from typing import Optional, List
from shutil import rmtree

from .base import BasePreProcessor


class NDVIPreprocessor(BasePreProcessor):
    """ Preprocesses the ESA CCI Landcover data """
    dataset = 'ndvi'

    def create_filename(self, netcdf_filepath: str,
                        subset_name: Optional[str] = None) -> str:
        """
        AVHRR-Land_v005_AVH13C1_NOAA-09_19860702_c20170612095548.nc
            =>
        1986_AVHRR-Land_v005_AVH13C1_NOAA-09_19860702_c20170612095548_kenya.nc
        """
        if netcdf_filepath[-3:] == '.nc':
            filename_stem = netcdf_filepath[:-3]
        else:
            filename_stem = netcdf_filepath

        year = filename_stem.split('_')[-2][:4]

        if subset_name is not None:
            new_filename = f"{year}_{filename_stem}_{subset_name}.nc"
        else:
            new_filename = f"{year}_{filename_stem}.nc"
        return new_filename

    def _preprocess_single(self, netcdf_filepath: Path,
                           subset_str: Optional[str] = 'kenya',
                           regrid: Optional[xr.Dataset] = None) -> None:
        """ Preprocess a single netcdf file (run in parallel if
        `parallel_processes` arg > 1)

        Process:
        -------
        * chop region of interset (ROI)
        * regrid to same spatial grid as a reference dataset (`regrid`)
        * create new dataset with these dimensions
        * Save the output file to new folder / filename

        Example filename:
        `AVHRR-Land_v005_AVH13C1_NOAA-19_20160826_c20170620152140.nc`
        """
        assert netcdf_filepath.name[-3:] == '.nc', \
            f'filepath name should be a .nc file. Currently: {netcdf_filepath.name}'

        print(f'Starting work on {netcdf_filepath.name}')
        try:
            ds = xr.open_dataset(netcdf_filepath)
            ds = ds.drop_dims(['ncrs', 'nv'])
        except ValueError:
            print(f'Unable to decode times for {netcdf_filepath.name}')
            ds = xr.open_dataset(netcdf_filepath, decode_times=False)
            # time = pd.to_datetime(ds.attrs['time_coverage_start'])
            time = pd.to_datetime(netcdf_filepath.name.split('_')[-2])
            ds.time.values = np.array([time])
            ds['latitude'] = np.linspace(-90, 90, 3600)
            ds['longitude'] = np.linspace(-180, 180, 7200)

        if 'latitude' in [d for d in ds.dims]:
            ds = ds.rename({'latitude': 'lat'})
        if 'longitude' in [d for d in ds.dims]:
            ds = ds.rename({'longitude': 'lon'})

        # 2. chop out EastAfrica
        if subset_str is not None:
            try:
                ds = self.chop_roi(ds, subset_str)
            except AssertionError:
                print("Trying regrid again with inverted latitude")
                ds = self.chop_roi(ds, subset_str, inverse_lat=True)

        # 3. regrid to same spatial resolution
        if regrid is not None:
            ds = self.regrid(ds, regrid, selected_var='NDVI')

        # 5. extract the ndvi data (reduce storage use)
        # NOTE: discarding the quality flags here
        ds = ds.NDVI.to_dataset(name='ndvi')

        # save to specific filename
        filename = self.create_filename(
            netcdf_filepath.name,
            subset_name=subset_str if subset_str is not None else None
        )
        print(f"Saving to {self.interim}/{filename}")
        try:
            ds.to_netcdf(self.interim / filename)
        except ValueError:
            ds.time.attrs = ''
            ds.to_netcdf(self.interim / filename)

        print(f"** Done for {self.dataset}: {filename} **")

    def preprocess(self, subset_str: Optional[str] = 'kenya',
                   regrid: Optional[Path] = None,
                   resample_time: Optional[str] = 'M',
                   upsampling: bool = False,
                   parallel_processes: int = 1,
                   years_to_process: Optional[List[int]] = None,
                   ignore_timesteps: Optional[List[str]] = None,
                   cleanup: bool = False) -> None:
        """Preprocess all of the NOAA NDVI .nc files to produce
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
        years: Optional[List[int]] = None,
            which subset of years to process data for (selected
            by using the years folders created by the NDVIExporter)
        ignore_timesteps: Optional[List[str]] = None
            Whether to remove any timesteps when merging dataset
            (i.e. if data is corrupt this requires manually specifying)
        cleanup: bool = True
            If true, delete interim files created by the class

        Note:
        ----
        - the raw data is downloaded at daily resolution and
        saved in annual directories.
        """
        print(f'Reading data from {self.raw_folder}. Writing to {self.interim}')
        nc_files = self.get_filepaths()

        # ability to only process particular years (for long jobs)
        if years_to_process is not None:
            assert np.isin(years_to_process, np.arange(1981, 2020)).all()
            yrs_strings = [str(y) for y in years_to_process]
            print(f'Running preprocess NDVI for: {yrs_strings}')
            nc_files = [
                f for f in nc_files
                if f.parents[0].name in yrs_strings
            ]

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
            upsampling=upsampling,
            ignore_timesteps=ignore_timesteps,
        )

        if cleanup:
            rmtree(self.interim)
