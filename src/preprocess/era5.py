from pathlib import Path
from functools import partial
import xarray as xr
import multiprocessing
from shutil import rmtree
import re
import numpy as np
import pandas as pd

from typing import Optional, List, Tuple

from .base import BasePreProcessor
from ..utils import get_modal_value_across_time


class ERA5MonthlyMeanPreprocessor(BasePreProcessor):

    dataset = "reanalysis-era5-single-levels-monthly-means"

    # some ERA5 variables need to be treated statically
    # they are recorded here
    static_vars = ["soil_type"]

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

        print(f"Done for ERA5 {netcdf_filepath.name}")

    def filter_outfiles(
        self, infiles: List[Path], filter_type: Optional[str] = None
    ) -> List[Path]:
        if filter_type is None:
            return infiles

        else:
            outfiles: List[Path] = []

            for filepath in infiles:
                is_static = False
                for var in self.static_vars:
                    if var in str(filepath):
                        is_static = True
                if (filter_type == "dynamic") and (not is_static):
                    outfiles.append(filepath)
                elif (filter_type == "static") and is_static:
                    outfiles.append(filepath)
        return outfiles

    def get_filepaths(
        self, folder: str = "raw", filter_type: Optional[str] = None
    ) -> List[Path]:
        """
        filter_type can be {None, 'static', 'dynamic'}
        """
        if folder == "raw":
            target_folder = self.raw_folder / self.dataset
        else:
            target_folder = self.interim
        outfiles = self.filter_outfiles(
            list(target_folder.glob("**/*.nc")), filter_type
        )
        outfiles.sort()

        return outfiles

    def merge_files(  # type: ignore
        self,
        subset_str: Optional[str] = "kenya",
        resample_time: Optional[str] = "M",
        upsampling: bool = False,
        filename: Optional[str] = None,
    ) -> Tuple[Optional[Path], ...]:

        # first, dynamic
        dynamic_filepaths = self.get_filepaths("interim", filter_type="dynamic")
        all_dyn_ds = []
        if len(dynamic_filepaths) > 0:
            ds_dyn = xr.open_mfdataset(
                dynamic_filepaths, combine="nested", concat_dim="time"
            )

            if resample_time is not None:
                ds_dyn = self.resample_time(ds_dyn, resample_time, upsampling)

            all_dyn_ds.append(ds_dyn)

            if filename is None:
                filename = (
                    f'data{"_" + subset_str if subset_str is not None else ""}.nc'
                )
            out_dyn = self.out_dir / filename

            ds_dyn.to_netcdf(out_dyn)
            print(f"\n**** {out_dyn} Created! ****\n")

        # then, static
        static_filepaths = self.get_filepaths("interim", filter_type="static")
        print(static_filepaths)
        if len(static_filepaths) > 0:
            ds_stat = xr.open_mfdataset(static_filepaths)

            da_list = []
            for var in ds_stat.data_vars:
                print(var)
                da_list.append(get_modal_value_across_time(ds_stat[var]))
            ds_stat_new = xr.merge(da_list)

            # one hot encode the soil-type if in dataset!
            if "slt" in list(ds_stat_new.data_vars):
                soil_type_legend = pd.DataFrame(
                    {
                        "code": [1, 2, 3, 4, 5, 6, 7],
                        "label_text": [
                            "slt_coarse",
                            "slt_medium",
                            "slt_medium_fine",
                            "slt_fine",
                            "slt_very_fine",
                            "slt_organic",
                            "slt_tropical_organic",
                        ],
                    }
                )

                ds_stat_new = self._one_hot_encode(
                    ds=ds_stat_new, legend=soil_type_legend, variable="slt"
                )

            output_folder = (
                self.preprocessed_folder / f"static/{self.dataset}_preprocessed"
            )
            if not output_folder.exists():
                output_folder.mkdir(exist_ok=True, parents=True)
            if filename is None:
                filename = (
                    f'data{"_" + subset_str if subset_str is not None else ""}.nc'
                )
            out_static: Optional[Path] = output_folder / filename

            # save legend if exists:
            if "slt" in list(ds_stat_new.data_vars):
                soil_type_legend.to_csv(output_folder / "legend.csv")

            ds_stat_new.to_netcdf(out_static)
            print(f"\n**** {out_static} Created! ****\n")
        else:
            out_static = None

        return out_dyn, out_static  # type: ignore

    def preprocess(
        self,
        subset_str: Optional[str] = "kenya",
        regrid: Optional[Path] = None,
        resample_time: Optional[str] = "M",
        upsampling: bool = False,
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
            If > 1, run the preprocessing in parallel with n_processes
        cleanup: bool = True
            If true, delete interim files created by the class
        """
        print(f"Reading data from {self.raw_folder}. Writing to {self.interim}")

        # get the filepaths for all of the downloaded data
        nc_files = self.get_filepaths()

        assert (
            nc_files != []
        ), f"No files found for {self.dataset}. Have you run the ERA5 Exporter?"

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

    def _one_hot_encode(
        self, ds: xr.Dataset, legend: pd.DataFrame, variable: str
    ) -> xr.Dataset:
        """
        Arguments:
        ---------
        : ds: xr.Dataset
            The dataset with the `variable` (str) we want to one
            hot encode.
        : legend: pd.DataFrame
            Dataframe with the code / label_text columns acting as a
            legend.
        : variable: str
            the variable in `ds` that we want to one hot encode.
        """
        assert all(
            np.isin(["code", "label_text"], legend.columns)
        ), "Need code / label_text columns in legend"
        assert variable in list(
            ds.data_vars
        ), f"expect {variable} to be in data vars: {list(ds.data_vars)}"

        # ensure categorical is type int
        # (prevents rounding errors)
        ds = ds.astype("int")

        for idx, row in legend.iterrows():
            value, label = row["code"], row["label_text"]
            ds[f"{label}_one_hot"] = (
                ds[variable].where(ds[variable] == value, 0).clip(min=0, max=1)
            )
        ds = ds.drop(variable)
        return ds


class ERA5HourlyPreprocessor(ERA5MonthlyMeanPreprocessor):
    dataset = "reanalysis-era5-single-levels"

    def merge_files(  # type: ignore
        self,
        subset_str: Optional[str] = "kenya",
        resample_time: Optional[str] = "W-MON",
        upsampling: bool = False,
        filename: Optional[str] = None,
    ) -> Tuple[Optional[Path], ...]:  # type: ignore

        # first, dynamic
        dynamic_filepaths = self.get_filepaths("interim", filter_type="dynamic")
        if len(dynamic_filepaths) > 0:
            _country_str = f"_{subset_str}.nc"  # '_[a-z]*.nc'
            variables = [
                re.sub(_country_str, "", p.name[8:]) for p in dynamic_filepaths
            ]

            # all_dyn_ds = []
            for variable in np.unique(variables):
                _dyn_fpaths = [
                    p
                    for p in dynamic_filepaths
                    if variable == re.sub(_country_str, "", p.name[8:])
                ]
                variable = "_".join(variable.split("_")[-2:])
                variable = f"hourly_{variable}"
                _ds_dyn = xr.open_mfdataset(_dyn_fpaths)
                # ds_dyn = xr.open_mfdataset(dynamic_filepaths)

                if resample_time is not None:
                    _ds_dyn = self.resample_time(_ds_dyn, resample_time, upsampling)

                filename_ = f'{variable}_data{"_" + subset_str if subset_str is not None else ""}.nc'
                out_dyn = self.out_dir / filename_

                # save to netcdf
                _ds_dyn.to_netcdf(out_dyn)
                print(f"\n**** {out_dyn} Created! ****\n")

                # all_dyn_ds.append(_ds_dyn)

            # too much data and gets killed (hourly data)
            # ds_dyn = xr.auto_combine(all_dyn_ds)

            # if filename is None:
            #     filename = (
            #         f'data{"_" + subset_str if subset_str is not None else ""}.nc'
            #     )
            # out_dyn = self.out_dir / filename

            # ds_dyn.to_netcdf(out_dyn)
            # print(f"\n**** {out_dyn} Created! ****\n")
        else:
            out_dyn = None  # type: ignore

        # then, static
        static_filepaths = self.get_filepaths("interim", filter_type="static")
        print(static_filepaths)
        if len(static_filepaths) > 0:
            ds_stat = xr.open_mfdataset(static_filepaths)

            da_list = []
            for var in ds_stat.data_vars:
                # get the modal value across time (collapse time)
                print(var)
                da_list.append(get_modal_value_across_time(ds_stat[var]))
            ds_stat_new = xr.merge(da_list)

            # one hot encode the soil-type if in dataset!
            if "slt" in list(ds_stat_new.data_vars):
                soil_type_legend = pd.DataFrame(
                    {
                        "code": [1, 2, 3, 4, 5, 6, 7],
                        "label_text": [
                            "slt_coarse",
                            "slt_medium",
                            "slt_medium_fine",
                            "slt_fine",
                            "slt_very_fine",
                            "slt_organic",
                            "slt_tropical_organic",
                        ],
                    }
                )

                ds_stat_new = self._one_hot_encode(
                    ds=ds_stat_new, legend=soil_type_legend, variable="slt"
                )

            # create folder
            output_folder = (
                self.preprocessed_folder / f"static/{self.dataset}_preprocessed"
            )
            if not output_folder.exists():
                output_folder.mkdir(exist_ok=True, parents=True)

            # create the filename
            if filename is None:
                filename = (
                    f'data{"_" + subset_str if subset_str is not None else ""}.nc'
                )
            out_static = output_folder / filename

            # save legend if exists:
            if "slt" in list(ds_stat_new.data_vars):
                soil_type_legend.to_csv(output_folder / "legend.csv")

            # save to netcdf
            ds_stat_new.to_netcdf(out_static)
            print(f"\n**** {out_static} Created! ****\n")
        else:
            out_static = None  # type: ignore

        return out_dyn, out_static
