import numpy as np
from pathlib import Path
import pandas as pd
import xarray as xr
from functools import partial
import multiprocessing
from shutil import rmtree
from typing import Optional, List, Tuple, cast

from ..base import BasePreProcessor
from .ouce_s5 import OuceS5Data


class S5Preprocessor(BasePreProcessor):
    dataset: str = "s5"

    def __init__(
        self,
        data_folder: Path = Path("data"),
        ouce_server: bool = False,
        n_processes: int = 1,
    ) -> None:
        super().__init__(data_folder)
        self.ouce_server = ouce_server
        self.n_processes = max(n_processes, 1)
        self.parallel = self.n_processes > 1

    def get_filepaths(  # type: ignore
        self, target_folder: Path, variable: str, grib: bool = True
    ) -> List[Path]:
        # if target_folder.name == 'raw':
        #     target_folder = self.raw_folder

        pattern = f"seasonal*/{variable}/*/*.grib" if grib else f"*/{variable}/*/*.nc"
        # get all files in */*/*
        outfiles = [f for f in target_folder.glob(pattern)]

        outfiles.sort()
        return outfiles

    @staticmethod
    def read_grib_file(filepath: Path) -> xr.Dataset:
        assert filepath.suffix in [
            ".grib",
            ".grb",
        ], f"This method is for \
        `grib` files. Not for {filepath.name}"
        ds = xr.open_dataset(filepath, engine="cfgrib")

        ds = ds.rename(
            {
                "time": "initialisation_date",
                "step": "forecast_horizon",
                "valid_time": "time",
            }
        )
        if ds.surface.values.size == 1:
            ds = ds.drop("surface")

        return ds

    @staticmethod
    def create_filename(
        filepath: Path,
        output_dir: Path,
        variable: str,
        regrid: Optional[xr.Dataset] = None,
        subset_name: Optional[str] = None,
    ) -> Path:
        # TODO: do we want each variable in separate folders / .nc files?
        subset_name = ("_" + subset_name) if subset_name is not None else ""
        filename = filepath.stem + f"_{variable}{subset_name}.nc"
        output_path = output_dir / variable / filename
        if not output_path.parents[0].exists():
            (output_path.parents[0]).mkdir(exist_ok=True, parents=True)
        return output_path

    def _preprocess(
        self,
        filepath: Path,
        subset_str: Optional[str] = None,
        regrid: Optional[xr.Dataset] = None,
        ouce_server: bool = False,
        **kwargs,
    ) -> Tuple[Path, str]:
        """preprocess a single s5 dataset (multi-variables per `.nc` file)"""
        print(f"\nWorking on {filepath.name}")

        if self.ouce_server:
            # undoes the preprocessing so that both are consistent
            infer = kwargs.pop("infer") if "infer" in kwargs.keys() else False
            # 1. read nc file
            ds = OuceS5Data().read_ouce_s5_data(filepath, infer=infer)
        else:  # downloaded from CDSAPI as .grib
            # 1. read grib file
            ds = self.read_grib_file(filepath)

        # find all variables (sometimes download multiple)
        coords = [c for c in ds.coords]
        vars = [v for v in ds.variables if v not in coords]
        variable = "-".join(vars)

        # 2. create the filepath and save to that location
        output_path = self.create_filename(
            filepath,
            self.interim,
            variable,
            subset_name=subset_str if subset_str is not None else None,
        )
        assert (
            output_path.name[-3:] == ".nc"
        ), f"\
        filepath name should be a .nc file. Currently: {filepath.name}"

        # IF THE FILE ALREADY EXISTS SKIP
        if output_path.exists():
            print(f"{output_path.name} already exists! Skipping.")
            return output_path, variable

        # 3. rename coords
        if "latitude" in coords:
            ds = ds.rename({"latitude": "lat"})
        if "longitude" in coords:
            ds = ds.rename({"longitude": "lon"})

        # 4. subset ROI
        if subset_str is not None:
            try:
                ds = self.chop_roi(ds, subset_str)
            except AssertionError:
                print("Retrying regridder with latitudes inverted")
                ds = self.chop_roi(ds, subset_str, inverse_lat=True)

        # 5. regrid (one variable at a time)
        if regrid is not None:
            assert all(
                np.isin(["lat", "lon"], [c for c in ds.coords])
            ), f"\
            Expecting `lat` `lon` to be in ds. dims : {[c for c in ds.coords]}"

            # regrid each variable individually
            all_vars = []
            for var in vars:
                if self.n_processes > 1:
                    # if parallel need to recreate new file each time
                    time = ds[var].time
                    d_ = self.regrid(
                        ds[var].to_dataset(name=var),
                        regrid,
                        clean=True,
                        reuse_weights=False,
                    )
                    d_ = d_.assign_coords(valid_time=time)
                else:
                    time = ds[var].time
                    d_ = self.regrid(
                        ds[var].to_dataset(name=var),
                        regrid,
                        clean=False,
                        reuse_weights=True,
                    )
                    d_ = d_.assign_coords(valid_time=time)
                all_vars.append(d_)
            # merge the variables into one dataset
            try:
                ds = xr.merge(all_vars).sortby("initialisation_date")
            except ValueError:
                ds = xr.merge(all_vars)

        if "initialisation_date" not in [d for d in ds.dims]:
            # add initialisation_date as a dimension
            ds = ds.expand_dims(dim="initialisation_date")

        # 6. save ds to output_path
        ds.to_netcdf(output_path)
        return output_path, variable

    def merge_all_interim_files(self, variable: str) -> xr.Dataset:
        # open all interim processed files (one variable)
        print("Reading and merging all interim .nc files")
        ds = xr.open_mfdataset((self.interim / variable).as_posix() + "/*.nc")
        ds = ds.sortby("initialisation_date")

        return ds

    # def resample_timesteps(self,
    #                        ds: xr.Dataset,
    #                        variable: str,
    #                        resample_str: Optional[str] = 'M',
    #                        upsampling: bool = False) -> xr.Dataset:
    #     # resample (NOTE: resample func removes the 'time' coord by default)
    #     print('Resampling the timesteps (initialisation_date)')
    #     if resample_str is not None:
    #         time = ds[variable].time
    #         ds = self.resample_time(
    #             ds, resample_str, upsampling, time_coord="time"
    #         )
    #         ds = ds.assign_coords(valid_time=time)

    #     return ds

    @staticmethod
    def _map_forecast_horizon_to_months_ahead(stacked: xr.Dataset) -> xr.Dataset:
        assert "forecast_horizon" in [c for c in stacked.coords], (
            "Expect the"
            "`stacked` dataset object to have `forecast_horizon` as a coord"
        )

        # map forecast horizons to months ahead
        map_ = {
            pd.Timedelta("28 days 00:00:00"): 1,
            pd.Timedelta("29 days 00:00:00"): 1,
            pd.Timedelta("30 days 00:00:00"): 1,
            pd.Timedelta("31 days 00:00:00"): 1,
            pd.Timedelta("59 days 00:00:00"): 2,
            pd.Timedelta("60 days 00:00:00"): 2,
            pd.Timedelta("61 days 00:00:00"): 2,
            pd.Timedelta("62 days 00:00:00"): 2,
            pd.Timedelta("89 days 00:00:00"): 3,
            pd.Timedelta("90 days 00:00:00"): 3,
            pd.Timedelta("91 days 00:00:00"): 3,
            pd.Timedelta("92 days 00:00:00"): 3,
        }

        fhs = [pd.Timedelta(fh) for fh in stacked.forecast_horizon.values]
        months = [map_[fh] for fh in fhs]
        stacked = stacked.assign_coords(months_ahead=("time", months))

        return stacked

    def stack_time(self, ds: xr.Dataset) -> xr.Dataset:
        """ Use the forecast horizon / initialisation date
        to create a dataset with 3 dimensions for subsetting
        the forecast data.

        Required to subset by the TRUE TIME (what time the forecast is of).
        I.e. a forecast initialised on 01-01-1994 for one month ahead has
        a TRUE TIME of 01-02-1994.

        e.g.
        IN:
            <xarray.Dataset>
            Dimensions:
            Coordinates:
            * initialisation_date  (initialisation_date)
            * forecast_horizon     (forecast_horizon)
            * lon                  (lon)
            * lat                  (lat)
              time                 (initialisation_date, forecast_horizon)
            Data variables:
                tprate               (initialisation_date, forecast_horizon, lat, lon)

        OUT:
            <xarray.Dataset>
            Dimensions:
            Coordinates:
            * lon                   (lon)
            * lat                   (lat)
            * time                  (time)
            * initialisation_dates  (initialisation_dates)
            * forecast_horizons     (forecast_horizons)
            Data variables:
                tprate                (lat, lon, time)

        Returns:
        -------
        ds: xr.Dataset
            dateset with a flattened time as an index

        initialisation_dates: np.array
            the intialisation dates as a flat numpy array

        forecast_horizons: np.array
            the forecast horizons as a flat numpy array
        """
        print("Stacking the [initialisation_date, forecast_horizon] coords")
        stacked = ds.stack(time=("initialisation_date", "forecast_horizon"))
        t = stacked.time.values

        # flatten the 2D time array [(timestamp, delta), ...]
        initialisation_dates = np.array(list(zip(*t))[0])
        forecast_horizons = np.array(list(zip(*t))[1])
        times = initialisation_dates + forecast_horizons

        # store as dimensions
        stacked["time"] = times
        stacked = stacked.assign_coords(
            initialisation_date=("time", initialisation_dates)
        )
        stacked = stacked.assign_coords(forecast_horizon=("time", forecast_horizons))
        if "valid_time" in [c for c in stacked.coords]:
            stacked = stacked.drop("valid_time")

        # remove all of the nan timesteps
        stacked = stacked.dropna(dim="time", how="all")

        # create months ahead coord
        stacked = self._map_forecast_horizon_to_months_ahead(stacked)

        return stacked

    @staticmethod
    def select_n_ensemble_members(ds: xr.Dataset, n: int = 25) -> xr.Dataset:
        """because the CDS data only has data for 25 ensemble members for the
        most recent dates (after 2000) there are a large number of nan values
        in the 26-51st ensemble members.

        Note: `number` dimension is the ensemble member

        Proportion of null values:
        array([0.        , 0.        , 0.        , 0.        , 0.        ,
               0.        , 0.        , 0.        , 0.        , 0.        ,
               0.        , 0.        , 0.        , 0.        , 0.        ,
               0.        , 0.        , 0.        , 0.        , 0.        ,
               0.        , 0.        , 0.        , 0.        , 0.        ,
               0.92307692, 0.92307692, 0.92307692, 0.92307692, 0.92307692,
               0.92307692, 0.92307692, 0.92307692, 0.92307692, 0.92307692,
               0.92307692, 0.92307692, 0.92307692, 0.92307692, 0.92307692,
               0.92307692, 0.92307692, 0.92307692, 0.92307692, 0.92307692,
               0.92307692, 0.92307692, 0.92307692, 0.92307692, 0.92307692,
               0.92307692])
        """
        ds = ds.isel(number=slice(0, n))
        return ds

    @staticmethod
    def get_n_timestep_ahead_data(
        ds: xr.Dataset, n_tstep: int, tstep_coord_name: str = "months_ahead"
    ) -> xr.Dataset:
        """ Get the data for the n timesteps ahead """
        assert tstep_coord_name in [c for c in ds.coords], (
            "expect the number of timesteps ahead to have been calculated"
            f" already. Coords: {[c for c in ds.coords]}"
        )

        variables = [v for v in ds.data_vars]
        all_nstep_list = []
        for var in variables:
            d_nstep = ds.loc[dict(time=ds[tstep_coord_name] == n_tstep)].rename(
                {var: var + f"_{n_tstep}"}
            )
            all_nstep_list.append(d_nstep)

        return xr.auto_combine(all_nstep_list)

    def create_variables_for_n_timesteps_predictions(
        self, ds: xr.Dataset, tstep_coord_name: str = "months_ahead"
    ) -> xr.Dataset:
        """Drop the forecast_horizon & initialisation_date variables"""
        assert all(
            np.isin(
                ["initialisation_date", "forecast_horizon", tstep_coord_name],
                [c for c in ds.coords],
            )
        ), (
            "Expecting to have "
            f"initialisation_date forecast_horizon {tstep_coord_name} in ds.coords"
            f"currently: {[c for c in ds.coords]}"
        )

        timesteps = np.unique(ds[tstep_coord_name])
        variables = [v for v in ds.data_vars]

        all_timesteps = []
        for step in timesteps:
            d = self.get_n_timestep_ahead_data(
                ds, step, tstep_coord_name=tstep_coord_name
            )
            d = d.drop(["initialisation_date", "forecast_horizon", tstep_coord_name])
            # drop the old variables too (so not duplicated)
            d = d.drop(variables)
            all_timesteps.append(d)

        return xr.auto_combine(all_timesteps)

    @staticmethod
    def get_variance_and_mean_over_number(ds: xr.Dataset) -> xr.Dataset:
        """Collapse the 'number' dimension (ensemble members) and return a
        Dataset with (lat, lon, time) coords and two variables:
        ['{var}_mean', '{var}_std']
        """
        variables = [v for v in ds.data_vars]

        # ensure that 'number' still exists in the coords
        assert "number" in [c for c in ds.coords], (
            "require `number` to "
            "be a coord in the Dataset object to collapse by mean/std"
        )

        # calculate mean and std collapsing the 'number' coordinate
        predict_ds_list = []
        for var in variables:
            print(f"Calculating the mean / std for forecast variable: {var}")
            mean_std = []
            mean_std.append(ds.mean(dim="number").rename({var: var + "_mean"}))
            mean_std.append(ds.std(dim="number").rename({var: var + "_std"}))
            predict_ds_list.append(xr.auto_combine(mean_std))

        return xr.auto_combine(predict_ds_list)

    def _process_interim_files(
        self,
        variables: List[str],
        resample_time: Optional[str] = "M",
        upsampling: bool = False,
        subset_str: Optional[str] = "kenya",
    ) -> None:
        # merge all of the preprocessed interim timesteps (../s5_interim/)
        for var in np.unique(variables):
            cast(str, var)
            ds = self.merge_all_interim_files(var)

            # remove 'time' from raw data (calculate from initialisation_date/forecast_horizon)
            if "time" in [c for c in ds.coords]:
                ds = ds.drop("time")

            # stack the timesteps to get a more standard dataset format
            # ('forecast_horizon', 'initialisation_date', 'lat', 'lon', 'number')
            # --> dims = ('lat', 'lon', 'time', 'number')
            ds = self.stack_time(ds)

            # select first 25 ensemble members (complete dataset)
            ds = self.select_n_ensemble_members(ds, n=25)

            # calculate mean/std over 'number'
            ds = self.get_variance_and_mean_over_number(ds)

            # calculate n_timestep ahead variables
            ds = self.create_variables_for_n_timesteps_predictions(ds)

            # resample time (N.B. if done before stacking time changes initialisation_date ...)
            if resample_time is not None:
                # print('WARNING: resampling time will alter the initialisation_dates')
                ds = self.resample_time(
                    ds=ds, resample_length=resample_time, upsampling=upsampling
                )

            # save to preprocessed netcdf
            out_path = self.out_dir / f"{self.dataset}_{var}_{subset_str}.nc"
            print(f"Saving data for variable: {var} \nto: {out_path}")
            ds.to_netcdf(out_path)

    def preprocess(
        self,
        variable: str,
        regrid: Optional[Path] = None,
        subset_str: Optional[str] = "kenya",
        resample_time: Optional[str] = "M",
        upsampling: bool = False,
        cleanup: bool = False,
        **kwargs,
    ) -> None:
        """Preprocesses the S5 data for all variables in the 'ds' file at once

        Argument:
        ---------
        subset_str: Optional[str] = 'kenya'
            whether to subset the data to a particular region

        regrid: Optional[Path] = None
            whether to regrid to the same lat/lon grid as the `regrid` ds

        resample_time: Optional[str] = None
            whether to resample the timesteps to a given `frequency` e.g. 'M'
            Coded differently to other preprocessors because the 'initialisation_date'
            which is stored initially as 'time' is important for calculating the 'valid_time'

        upsampling: bool = False
            are you upsampling the time frequency (e.g. monthly -> daily)

        variable: str
            Which variable do we want to process? [NOTE: each call
            to `obj.preprocess()` preprocesses ONE variable]

            if self.ouce_server then also require a variable string
            to build the filepath to the data to preprocess

        cleanup: bool = False
            Whether to cleanup the self.interim directory

        kwargs: dict
            keyword arguments (mostly for pytest!)
            'ouce_dir' : Path - the test directory to use for reading .nc files
            'infer' : bool - whether to infer the frequency when creating
                the forecast horizon for ouce_data.
        """
        if self.ouce_server:
            # data already in netcdf but needs other preprocessing
            assert (
                variable is not None
            ), f"Must pass a variable argument when\
            preprocessing the S5 data on the OUCE servers"
            os = OuceS5Data()

            # get kwargs
            ouce_dir = kwargs.pop("ouce_dir") if "ouce_dir" in kwargs.keys() else None

            filepaths = os.get_ouce_filepaths(variable=variable, parent_dir=ouce_dir)
        else:
            filepaths = self.get_filepaths(
                target_folder=self.raw_folder, variable=variable
            )

        # argument needs the regrid file
        if regrid is not None:
            regrid_ds = xr.open_dataset(regrid)
        else:
            regrid_ds = None

        if not self.parallel:
            out_paths = []
            variables = []
            for filepath in filepaths:
                output_path, variable = self._preprocess(
                    filepath=filepath,
                    subset_str=subset_str,
                    regrid=regrid_ds,
                    ouce_server=self.ouce_server,
                    **kwargs,
                )
                out_paths.append(output_path)
                variables.append(variable)

        else:
            # Not implemented parallel yet
            pool = multiprocessing.Pool(processes=5)
            outputs = pool.map(
                partial(
                    self._preprocess,
                    ouce_server=self.ouce_server,
                    subset_str=subset_str,
                    regrid=regrid,
                ),
                filepaths,
            )
            print("\nOutputs (errors):\n\t", outputs)

        # process the interim files (each timestep)
        self._process_interim_files(
            variables,
            subset_str=subset_str,
            resample_time=resample_time,
            upsampling=upsampling,
        )

        if cleanup:
            rmtree(self.interim)
