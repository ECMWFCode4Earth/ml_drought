from pathlib import Path
import xarray as xr
import multiprocessing
from functools import partial
from typing import Optional, List
from shutil import rmtree

from .base import BasePreProcessor


class ERA5LandPreprocessor(BasePreProcessor):
    """ Preprocesses the ERA5 Land data """

    dataset = "reanalysis-era5-land"

    @staticmethod
    def create_filename(
        netcdf_filepath: Path, subset_name: Optional[str] = None
    ) -> str:

        var_name = netcdf_filepath.parts[-3]
        months = netcdf_filepath.parts[-1][:-3]
        year = netcdf_filepath.parts[-2]

        stem = f"{year}_{months}_{var_name}"
        if subset_name is not None:
            stem = f"{stem}_{subset_name}"
        return f"{stem}.nc"

    def _preprocess_single(
        self,
        netcdf_filepath: Path,
        subset_str: Optional[str] = "kenya",
        regrid: Optional[xr.Dataset] = None,
    ) -> None:
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
        print(f"Processing {netcdf_filepath.name}")

        # 1. read in the dataset
        ds = xr.open_dataset(netcdf_filepath).rename(
            {"longitude": "lon", "latitude": "lat"}
        )

        # 2. chop out EastAfrica
        if subset_str is not None:
            ds = self.chop_roi(ds, subset_str, inverse_lat=True)

        if regrid is not None:
            ds = self.regrid(ds, regrid)

        filename = self.create_filename(
            netcdf_filepath, subset_name=subset_str if subset_str is not None else None
        )
        print(f"Saving to {self.interim}/{filename}")
        ds.to_netcdf(self.interim / filename)

        print(f"Done for ERA5-Land {netcdf_filepath.name}")

    def preprocess(
        self,
        subset_str: Optional[str] = "kenya",
        regrid: Optional[Path] = None,
        resample_time: Optional[str] = "M",
        upsampling: bool = False,
        parallel_processes: int = 1,
        variable: Optional[str] = None,
        years: Optional[List[int]] = None,
        cleanup: bool = True,
    ) -> None:
        """Preprocess all of the ERA5-Land .nc files to produce
        one subset file.
        Arguments
        ----------
        :param: subset_str: Optional[str] = 'kenya'
            Whether to subset Kenya when preprocessing
        :param: regrid: Optional[Path] = None
            If a Path is passed, the CHIRPS files will be regridded to have the same
            grid as the dataset at that Path. If None, no regridding happens
        :param: resample_time: str = 'M'
            If not None, defines the time length to which the data will be resampled
        :param: upsampling: bool = False
            If true, tells the class the time-sampling will be upsampling. In this case,
            nearest instead of mean is used for the resampling
        :param: variable: Optional[str] = None
            the variable that you want to preprocess. If None then will
            process ALL variables that have been downloaded to the
            `data/raw/reanalysis-era5-land` by the ERA5LandExporter
        :param: parallel_processes: int = 1
            If > 1, run the preprocessing in parallel
        :param: years: Optional[List[int]] = None
            preprocess a subset of the years from the raw data
        :param: cleanup: bool = True
            If true, delete interim files created by the class

        Note:
        ----
        - the raw data is downloaded at annual resolution by default
        """
        print(f"Reading data from {self.raw_folder}. Writing to {self.interim}")
        nc_files = self.get_filepaths()
        if years is not None:
            nc_files = [
                f for f in nc_files if int(f.parents[0]) in years  # type: ignore
            ]

        # run for one variable or all variables?
        if variable is not None:
            variables = [d.name for d in (self.raw_folder / self.dataset).iterdir()]
            assert variable in variables, (
                "Expect the variable provided" f"to be in {variables}"
            )
            print(f"Running preprocessor for var: {variable}")
            nc_files = [f for f in nc_files if f.parents[1].name == variable]

        if regrid is not None:
            regrid = self.load_reference_grid(regrid)

        # parallel processing ?
        if parallel_processes <= 1:  # sequential
            for file in nc_files:
                self._preprocess_single(file, subset_str, regrid)
        else:
            pool = multiprocessing.Pool(processes=parallel_processes)
            outputs = pool.map(
                partial(self._preprocess_single, subset_str=subset_str, regrid=regrid),
                nc_files,
            )
            print("\nOutputs (errors):\n\t", outputs)

        # merge and resample files
        filename = (
            f'{variable}_data{"_" + subset_str if subset_str is not None else ""}.nc'
        )
        self.merge_files(
            subset_str=subset_str,
            resample_time=resample_time,
            upsampling=upsampling,
            filename=filename,
        )

        if cleanup:
            rmtree(self.interim)


class ERA5LandMonthlyMeansPreprocessor(ERA5LandPreprocessor):
    """Process the ERA5-Land monthly statistics data"""

    dataset = "reanalysis-era5-land-monthly-means"
