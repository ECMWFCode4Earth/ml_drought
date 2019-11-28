"""
Clement Atzberger data (from the BOKU Life Science group)
---------------------------------------------------------
clement.atzberger@boku.ac.at

Paper outlining the preprocessing already done
-----------------------------------------------
https://www.mdpi.com/2072-4292/8/4/267
Klisch, A.; Atzberger, C. Operational Drought Monitoring in
Kenya Using MODIS NDVI Time Series. Remote Sens. 2016, 8, 267.
"""
from pathlib import Path
from functools import partial
import xarray as xr
import multiprocessing
from shutil import rmtree
from typing import Optional

from .base import BasePreProcessor


class MODISNDVIPreprocessor(BasePreProcessor):
    """ Preprocesses the MODIS NDVI data """

    dataset = 'modis_ndvi'

    def __init__(self, data_folder: Path = Path('data'), resolution: str = '1000') -> None:
        assert str(resolution) in ['250', '1000'], 'resolution needs to be one of' \
            f'[250, 1000]m resolution. You provided: {resolution}'
        self.dataset = self.dataset + f'_{str(resolution)}'

        super().__init__(data_folder)

    def _preprocess_single(self, netcdf_filepath: Path,
                           subset_str: Optional[str] = 'kenya',
                           regrid: Optional[xr.Dataset] = None) -> None:
        """Run the Preprocessing steps for the CHIRPS data

        Process:
        -------
        * assign time stamp
        * assign lat lon
        * create new dataset with these dimensions
        * Save the output file to new folder
        """
        print(f'Starting work on {netcdf_filepath.name}')
        # 1. read in the dataset
        ds = xr.open_dataset(netcdf_filepath)

        # 2. chop out ROI
        if subset_str is not None:
            ds = self.chop_roi(ds, subset_str)

        if regrid is not None:
            ds = self.regrid(ds, regrid)

        # 6. create the filepath and save to that location
        assert netcdf_filepath.name[-3:] == '.nc', \
            f'filepath name should be a .nc file. Currently: {netcdf_filepath.name}'

        filename = self.create_filename(
            netcdf_filepath.name,
            subset_name=subset_str if subset_str is not None else None
        )
        print(f"Saving to {self.interim}/{filename}")
        ds.to_netcdf(self.interim / filename)

        print(f"** Done for MODIS NDVI {netcdf_filepath.name} **")

    @staticmethod
    def create_filename(netcdf_filepath: str,
                        subset_name: Optional[str] = None) -> str:
        """
        MCD13A2.t201806.006.EAv1.1_km_10_days_NDVI.O1.nc
            =>
        MCD13A2.t201806.006.EAv1.1_km_10_days_NDVI.O1_kenya.nc
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

    def preprocess(self,
                   subset_str: Optional[str] = 'kenya',
                   regrid: Optional[Path] = None,
                   n_parallel_processes: int = 1,
                   resample_time: Optional[str] = 'M',
                   upsampling: bool = False,
                   cleanup: bool = True) -> None:
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

        print(f"Reading data from {self.raw_folder}. \
            Writing to {self.interim}")
        n_parallel_processes = max(n_parallel_processes, 1)
        if n_parallel_processes > 1:
            pool = multiprocessing.Pool(processes=n_parallel_processes)
            outputs = pool.map(partial(self._preprocess_single,
                                       subset_str=subset_str,
                                       regrid=regrid), nc_files)
            errors = [o for o in outputs if not isinstance(o, Path)]

            # TODO check how these errors are being saved (now all paths returned)
            self.print_output(outputs)
            self.save_errors(errors)
        else:
            for file in nc_files:
                self._preprocess_single(
                    file, subset_str=subset_str, regrid=regrid)

        self.merge_files(subset_str=subset_str, resample_time=resample_time,
                         upsampling=upsampling)
        if cleanup:
            rmtree(self.interim)
