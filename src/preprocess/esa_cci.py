from pathlib import Path
import xarray as xr
import pandas as pd
import multiprocessing
from functools import partial
from typing import Optional, List
from shutil import rmtree

from .base import BasePreProcessor


class ESACCIPreprocessor(BasePreProcessor):
    """ Preprocesses the ESA CCI Landcover data """
    dataset = 'esa_cci_landcover'

    def create_filename(self, netcdf_filepath: str,
                        subset_name: Optional[str] = None) -> str:
        """
        ESACCI-LC-L4-LCCS-Map-300m-P1Y-1992-v2.0.7b.nc
            =>
        ESACCI-LC-L4-LCCS-Map-300m-P1Y-1992-v2.0.7b_kenya.nc
        """
        if netcdf_filepath[-3:] == '.nc':
            filename_stem = netcdf_filepath[:-3]
        else:
            filename_stem = netcdf_filepath

        year = filename_stem.split('-')[-2]

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
        * assign time stamp
        * Save the output file to new folder
        """
        assert netcdf_filepath.name[-3:] == '.nc', \
            f'filepath name should be a .nc file. Currently: {netcdf_filepath.name}'

        print(f'Starting work on {netcdf_filepath.name}')
        ds = xr.open_dataset(netcdf_filepath)

        # 2. chop out EastAfrica
        if subset_str is not None:
            try:
                ds = self.chop_roi(ds, subset_str)
            except AssertionError:
                print("Trying regrid again with inverted latitude")
                ds = self.chop_roi(ds, subset_str, inverse_lat=True)

        # 3. regrid to same spatial resolution ...?
        if regrid is not None:
            ds = self.regrid(ds, regrid)

        # 4. assign time stamp
        try:  # try inferring from the ds.attrs
            time = pd.to_datetime(ds.attrs['time_coverage_start'])
        except KeyError:  # else infer from filename (for tests)
            year = netcdf_filepath.name.split('-')[-2]
            time = pd.to_datetime(f'{year}-01-01')

        ds = ds.assign_coords(time=time)
        ds = ds.expand_dims('time')

        # 5. extract the landcover data (reduce storage use)
        ds = ds.lccs_class.to_dataset(name='lc_class')

        # save to specific filename
        filename = self.create_filename(
            netcdf_filepath.name,
            subset_name=subset_str if subset_str is not None else None
        )
        print(f"Saving to {self.interim}/{filename}")
        ds.to_netcdf(self.interim / filename)

        print(f"** Done for ESA CCI landcover: {filename} **")

    def preprocess(self, subset_str: Optional[str] = 'kenya',
                   regrid: Optional[Path] = None,
                   resample_time: Optional[str] = 'M',
                   upsampling: bool = True,
                   parallel_processes: int = 1,
                   years: Optional[List[int]] = None,
                   cleanup: bool = True) -> None:
        """Preprocess all of the ESA CCI landcover .nc files to produce
        one subset file resampled to the timestep of interest.
        (downloaded as annual timesteps)

        Note:
        ----
        - because the landcover data only goes back to 1993 for all dates
        before 1993 that we need data for  we have selected the `modal`
        class from the whole data range (1993-2019).
        - This assumes that landcover is relatively consistent in the 1980s
        as the 1990s, 2000s and 2010s
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

        self.merge_files(subset_str, resample_time, upsampling)

        if cleanup:
            rmtree(self.interim)
