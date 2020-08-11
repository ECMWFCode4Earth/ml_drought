from pathlib import Path
from functools import partial
import xarray as xr
import numpy as np
import multiprocessing
from shutil import rmtree

from typing import Dict, Tuple, List, Optional

from .base import BasePreProcessor


class PlanetOSPreprocessor(BasePreProcessor):

    dataset = "era5POS"

    @staticmethod
    def _rename_time_dim(df: xr.Dataset) -> xr.Dataset:

        dims = [x for x in df.dims if x != "nv"]
        for dim in dims:
            if "time" in dim:
                df = df.rename({dim: "time"})
        return df

    @staticmethod
    def _rotate_and_filter(ds: xr.Dataset) -> xr.Dataset:
        # rotates the longitudes so they are in the -180 to 180 range
        data_vars = [x for x in ds.data_vars if x != "time1_bounds"]
        dims = ["time", "lat", "lon"]
        _time = (int(ds["time.year"].min().values), int(ds["time.month"].min().values))

        print(f"Rotating {data_vars} on {dims} for {_time}")

        boundary = sum(ds.lon.values < 180)
        new_lon = np.concatenate(
            [ds.lon.values[boundary:] - 360, ds.lon.values[:boundary]]
        )

        dataarrays: Dict[str, Tuple[List[str], np.ndarray]] = {}

        for var in data_vars:
            var_array = ds[var]
            upper, lower = var_array[:, :, :boundary], var_array[:, :, boundary:]
            rotated_var = np.concatenate((lower, upper), axis=-1)

            dataarrays[var] = (dims, rotated_var)

        return xr.Dataset(
            dataarrays,
            coords={"lat": ds.lat.values, "lon": new_lon, "time": ds.time.values},
        )

    @staticmethod
    def create_filename(
        netcdf_filepath: Path, subset_name: Optional[str] = None
    ) -> str:

        var_name = netcdf_filepath.name[:-3]
        month = netcdf_filepath.parts[-2]
        year = netcdf_filepath.parts[-3]

        stem = f"{year}_{month}_{var_name}"
        if subset_name is not None:
            stem = f"{stem}_{subset_name}"
        return f"{stem}.nc"

    def _preprocess_single(
        self,
        netcdf_filepath: Path,
        subset_str: Optional[str] = "kenya",
        regrid: Optional[xr.Dataset] = None,
    ) -> None:

        print(f"Processing {netcdf_filepath.name}")
        # 1. read in the dataset
        ds = self._rename_time_dim(xr.open_dataset(netcdf_filepath))
        ds = self._rotate_and_filter(ds)

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

        print(f"Done for ERA5 Planet OS {netcdf_filepath.name}")

    def preprocess(
        self,
        subset_str: Optional[str] = "kenya",
        regrid: Optional[Path] = None,
        resample_time: Optional[str] = "M",
        upsampling: bool = False,
        # parallel: bool = False,
        n_processes: int = 1,
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
        n_processes: int = 1
            If > 1, run the preprocessing with n_processes
        cleanup: bool = True
            If true, delete interim files created by the class
        """
        print(f"Reading data from {self.raw_folder}. Writing to {self.interim}")

        # get the filepaths for all of the downloaded data
        nc_files = self.get_filepaths()

        if regrid is not None:
            regrid = self.load_reference_grid(regrid)

        n_processes = max(n_processes, 1)
        if n_processes > 1:
            pool = multiprocessing.Pool(processes=n_processes)

            outputs = pool.map(
                partial(self._preprocess_single, subset_str=subset_str, regrid=regrid),
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
