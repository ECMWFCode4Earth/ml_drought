from pathlib import Path
import xarray as xr
import pandas as pd
import multiprocessing
from functools import partial
from typing import Optional, List, Dict
from shutil import rmtree
import numpy as np
from numpy import copy

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

    def remap_values_to_even_spaced_integers(self, ds: xr.Dataset,
                                             remap: Optional[Dict] = None) -> xr.Dataset:
        """https://stackoverflow.com/a/3404089/9940782"""
        assert 'lc_class' in ds.data_vars, "Should be run after preprocessing!"
        new_ds = xr.ones_like(ds)

        # extract and copy numpy array
        array = ds.lc_class.values
        new_array = copy(array)

        if not remap:
            # create remap dictionary
            legend = pd.read_csv(self.raw_folder / self.dataset / 'legend.csv')
            valid_vals = legend.code.values
            remap_vals = np.arange(0, (len(valid_vals) * 10), 10)

            assert isinstance(valid_vals, np.ndarray)
            remap = dict(zip(valid_vals, remap_vals))

            # save the new codes to pandas dataframe
            legend['new_code'] = remap_vals
            legend.to_csv(self.out_dir / 'legend.csv')

        # perform the remap
        for k, v in remap.items(): new_array[array == k] = v

        # reassign values to new ds
        new_ds['lc_class'] = (['time', 'lat', 'lon'], new_array)

        return new_ds

    def _preprocess_single(self, netcdf_filepath: Path,
                           subset_str: Optional[str] = 'kenya',
                           regrid: Optional[xr.Dataset] = None,
                           remap_dict: Optional[Dict] = None) -> None:
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
        # NOTE: have to remove the extra vars for the regridder
        ds = ds.drop([
            'processed_flag', 'current_pixel_state',
            'observation_count', 'change_count', 'crs'
        ])
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

        # 6. remap values
        ds = self.remap_values_to_even_spaced_integers(ds, remap_dict)

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
                   cleanup: bool = True,
                   remap_dict: Optional[Dict] = None) -> None:
        """Preprocess all of the ESA CCI landcover .nc files to produce
        one subset file resampled to the timestep of interest.
        (downloaded as annual timesteps)

        Arguments:
        ---------

        remap_dict: Optional[Dict] = None
            provide a dictionary to manually remap the values (for pytest)

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
                self._preprocess_single(file, subset_str, regrid, remap_dict)
        else:
            pool = multiprocessing.Pool(processes=parallel_processes)
            outputs = pool.map(
                partial(self._preprocess_single,
                        subset_str=subset_str,
                        regrid=regrid, remap_dict=remap_dict),
                nc_files)
            print("\nOutputs (errors):\n\t", outputs)

        self.merge_files(subset_str, resample_time, upsampling)

        if cleanup:
            rmtree(self.interim)
