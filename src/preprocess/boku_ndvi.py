from pathlib import Path
from shutil import rmtree
import xarray as xr
import os

from typing import Optional

from .base import BasePreProcessor


class BokuNDVIPreprocessor(BasePreProcessor):
    def preprocess(
        self,
        subset_str: Optional[str] = "kenya",
        regrid: Optional[Path] = None,
        resample_time: Optional[str] = "M",
        upsampling: bool = False,
        parallel: bool = False,
        cleanup: bool = True,
    ) -> None:
        """ Preprocess all of the era5 POS .nc files to produce
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
        cleanup: bool = True
            If true, delete interim files created by the class
        """
        print(
            f"Reading data from {self.raw_folder}. Writing to {self.interim}")

        # get the filepaths for all of the downloaded data
        nc_files = self.get_filepaths()

        if regrid is not None:
            regrid = self.load_reference_grid(regrid)

        if parallel:
            pool = multiprocessing.Pool(processes=100)
            outputs = pool.map(
                partial(self._preprocess_single,
                        subset_str=subset_str, regrid=regrid),
                nc_files,
            )
            print("\nOutputs (errors):\n\t", outputs)
        else:
            for file in nc_files:
                self._preprocess_single(file, subset_str, regrid)

        # merge all of the timesteps
        self.merge_files(subset_str, resample_time, upsampling)

        if cleanup:
            rmtree(self.interim)



class BokuNDVI1000Preprocessor(BokuNDVIPreprocessor):
    # 1km pixel
    dataset: str = 'boku_ndvi_1000'
    static = False
    resolution: str = '1000'


class BokuNDVI250Preprocessor(BokuNDVIPreprocessor):
    # 250m pixel
    dataset: str = 'boku_ndvi_250'
    static = False
    resolution: str = '250'
