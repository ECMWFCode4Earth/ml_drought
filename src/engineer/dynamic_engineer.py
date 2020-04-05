from pathlib import Path
import xarray as xr
import numpy as np
from collections import defaultdict
import pickle
import pandas as pd
from pandas.tseries.offsets import Day
from typing import DefaultDict, Dict, Tuple, Optional, Union, List, Any
from collections import Iterable

from src.engineer.base import _EngineerBase
from scripts.utils import _rename_directory


class DynamicEngineer(_EngineerBase):
    """Engineer the data for use by the DynamicDataLoader.

    Major differences:
    - save one dynamic_ds with all dynamic data -> (features/{exp}/data.nc)
    - save one static_ds -> (features/static/data.nc)
    - Do more preprocessing steps here
        a. drop the ignore vars (reduce memory cost)
        b. augment the static data here
        c. calculate log of predictand
    """

    name: str = "one_timestep_forecast"

    dynamic_ignore_vars: List[str]
    static_ignore_vars: List[str]

    def _calculate_normalization_dict(
        self, train_ds: xr.Dataset, static: bool, latlon: bool = True
    ) -> DefaultDict[str, Dict[str, float]]:
        """calculate a dictionary storing the mean and std values
        for each of the variables in `train_ds`"""
        normalization_dict: DefaultDict[str, Dict[str, float]] = defaultdict(dict)

        if static:
            dims = [c for c in train_ds.coords]
        else:
            if latlon:
                dims = ["lat", "lon", "time"]
            else:  # ASSUME 1D
                assert (
                    len([c for c in train_ds.coords if c != "time"]) == 1
                ), "Only works with one dimension"
                dimension_name = [c for c in train_ds.coords][0]
                dims = [dimension_name, "time"]

        for var in train_ds.data_vars:
            if var.endswith("one_hot"):
                mean = 0
                std = 1

            mean = float(train_ds[var].mean(dim=dims, skipna=True).values)
            std = float(train_ds[var].std(dim=dims, skipna=True).values)
            normalization_dict[var]["mean"] = mean
            normalization_dict[var]["std"] = std

        return normalization_dict

    def save_ds(self, ds: xr.Dataset, static: bool) -> None:
        if static:
            ds.to_netcdf(self.static_output_folder / "data.nc")
        else:
            ds.to_netcdf(self.output_folder / "data.nc")

    def save_normalization_dict(
        self, normalization_dict: DefaultDict[str, Dict[str, float]], static: bool
    ):
        if static:
            savepath = self.static_output_folder / "normalizing_dict.pkl"
        else:
            savepath = self.output_folder / "normalizing_dict.pkl"
        with savepath.open("wb") as f:
            pickle.dump(normalization_dict, f)

    @staticmethod
    def get_max_train_date(ds, test_year: Union[str, List[str]]) -> Tuple[pd.Timestamp]:
        """"""
        # get the minimum test_year
        if isinstance(test_year, Iterable):
            test_year = min([int(y) for y in test_year])

        # because of time-series nature of data
        # ASSUMPTION: only train on timesteps before the minimum test-date
        ds = ds.sortby("time")
        min_test_date = pd.to_datetime(f"{test_year}-01-01")
        max_train_date = min_test_date - Day(1)
        min_ds_date = pd.to_datetime(ds.time.min().values)

        return min_test_date, max_train_date, min_ds_date

    def create_normalization_dict(
        self,
        ds: xr.Dataset,
        static: bool,
        test_year: Optional[Union[List[str], str]] = None,
        latlon: bool = False,
    ) -> DefaultDict[str, Dict[str, float]]:
        """save the dynamic or static normalization dict"""
        if static:
            normalization_dict = self._calculate_normalization_dict(
                ds, latlon=False, static=True
            )
        else:
            assert test_year is not None, (
                "Must provide the test years for the calculation"
                " of the dynamic normalisation dict"
            )
            assert "time" in [d for d in ds.dims], "Expecting to have `time` dimension"

            min_test_date, max_train_date, min_ds_date = self.get_max_train_date(
                ds, test_year
            )
            train_ds = ds.sel(time=slice(min_ds_date, max_train_date))

            normalization_dict = self._calculate_normalization_dict(
                train_ds, latlon=False, static=static
            )

        # save the normalization dict so there is a copy
        self.save_normalization_dict(
            static=static, normalization_dict=normalization_dict
        )

        return normalization_dict

    def engineer_static(
        self,
        latlon: bool = False,
        augment_static: bool = False,
        dynamic_ds: Optional[xr.Dataset] = None,
        test_years: Optional[Union[List[str], str]] = None,
        static_ignore_vars: Optional[List[str]] = None,
        variables_for_ohe: Optional[List[str]] = None,
    ) -> None:
        self.check_and_move_already_existing_nc_file(self.static_output_folder)

        print("Engineering the static data\n" f"\tAugment: {augment_static}")

        # load static data from interim/{*}_preprocessed/
        if augment_static:
            assert (dynamic_ds is not None) & (test_years is not None), (
                "If want to augment static then require dynamic_ds and"
                " test_years to prevent leakage of information"
            )
            assert (
                False
            ), "Finish this off by dropping the values and calling the augment function"
            dynamic_ds = dynamic_ds

        # engineer the static data
        static_ds = self._make_dataset(static=True, latlon=latlon)

        # TODO: remove this hack
        # ensure that station_id is 'int'
        if "station_id" in list(static_ds.dims):
            static_ds["station_id"] = static_ds["station_id"].astype(np.dtype("int64"))
            static_ds = static_ds.sortby("station_id")

        # ONE HOT ENCODE features
        if variables_for_ohe is not None:
            variables_for_ohe = [
                v
                for v in variables_for_ohe
                if (v not in [ign for ign in static_ignore_vars])
                & (v in List(static_ds.data_vars))
            ]
            if variables_for_ohe != []:
                static_ds = self.one_hot_encode_vars(
                    static_ds, variables_for_ohe, static=True
                )

        # drop the ignore vars
        static_ds = self._drop_ignore_vars(static_ds, ignore_vars=static_ignore_vars)

        # calculate normalization dict
        static_normalization_dict = self.create_normalization_dict(
            static_ds, static=True, latlon=latlon
        )

        # save static_ds to `features/static/data.nc`
        static_ds.to_netcdf(self.static_output_folder / "data.nc")

    @staticmethod
    def _drop_ignore_vars(
        ds: xr.Dataset, ignore_vars: Optional[List[str]] = None
    ) -> xr.Dataset:
        if ignore_vars is not None:
            #  only include the vars in ignore_vars that are in x.data_vars
            ignore_vars = [
                v for v in ignore_vars if v in [var_ for var_ in ds.data_vars]
            ]
            ds = ds.drop(ignore_vars)

        return ds

    @staticmethod
    def check_and_move_already_existing_nc_file(directory: Path) -> None:
        """Check for data.nc in the engineer folder and move it"""
        if (directory / "data.nc").exists():
            print("Require the engineered folder to be empty. Moving data.nc")
            _rename_directory(
                directory / "data.nc", directory / "data.nc_", with_datetime=True
            )

    def engineer_dynamic(
        self,
        test_years: Union[List[str], str],
        latlon: bool = False,
        logy: bool = False,
        target_variable: str = "discharge_spec",
        dynamic_ignore_vars: Optional[List[str]] = None,
    ) -> xr.Dataset:
        self.check_and_move_already_existing_nc_file(self.output_folder)

        print(
            "Engineering the dynamic data\n"
            f"\tTarget Var: {target_variable}\n"
            f"\tLogy: {logy}\n"
            f"\tIgnore Vars: {dynamic_ignore_vars}\n"
        )
        # pretty much move from interim/ -> features/
        dynamic_ds = self._make_dataset(static=False, latlon=latlon)

        # TODO: remove this hack
        # ensure that station_id is 'int'
        if "station_id" in list(dynamic_ds.dims):
            dynamic_ds["station_id"] = dynamic_ds["station_id"].astype(
                np.dtype("int64")
            )
            dynamic_ds = dynamic_ds.sortby("station_id")

        # ensure that time is datetime64 dtype
        if dynamic_ds["time"].dtype != np.dtype("datetime64[ns]"):
            dynamic_ds["time"] = dynamic_ds["time"].astype(np.dtype("datetime64[ns]"))

        # log the y
        if logy:
            dynamic_ds["target_var_original"] = dynamic_ds[target_variable]
            dynamic_ds[target_variable] = np.log(dynamic_ds[target_variable] + 0.001)

        # drop the ignore vars
        # NB: cannot ignore the target var here because need for creating
        # xy pairs in the DynamicDataLoader
        dynamic_ignore_vars = [v for v in dynamic_ignore_vars if v != target_variable]

        dynamic_ds = self._drop_ignore_vars(dynamic_ds, ignore_vars=dynamic_ignore_vars)

        # calculate the normalization dict
        dynamic_normalization_dict = self.create_normalization_dict(
            dynamic_ds, static=False, test_year=test_years, latlon=latlon
        )

        # save to `data.nc`
        dynamic_ds.to_netcdf(self.output_folder / "data.nc")

        return dynamic_ds

    def engineer(
        self,
        test_years: Union[List[str], str],
        target_variable: str = "discharge_spec",
        augment_static: bool = False,
        logy: bool = False,
        static_ignore_vars: Optional[List[str]] = None,
        dynamic_ignore_vars: Optional[List[str]] = None,
        latlon: bool = False,
        variables_for_ohe: Optional[List[str]] = None,
    ) -> None:
        dynamic_ds = self.engineer_dynamic(
            test_years=test_years,
            latlon=latlon,
            target_variable=target_variable,
            dynamic_ignore_vars=dynamic_ignore_vars,
            logy=logy,
        )

        if augment_static:
            self.engineer_static(
                latlon=latlon,
                augment_static=True,
                dynamic_ds=dynamic_ds,
                static_ignore_vars=static_ignore_vars,
                variables_for_ohe=variables_for_ohe,
            )
        else:
            self.engineer_static(
                latlon=latlon,
                augment_static=False,
                static_ignore_vars=static_ignore_vars,
            )

    def augment_static_data(
        dynamic_ds: xr.Dataset,
        static_ds: xr.Dataset,
        test_year: Optional[List[str]] = None,
        dynamic_ignore_vars: List[str] = None,
        global_means: bool = True,
        spatial_means: bool = True,
    ) -> xr.Dataset:
        """ Create our own aggregations from the dynamic data

        NOTE: unnecessary for CAMELS because this data can
        just be taken from pre-computed means
        """
        # get the minimum test_year
        if isinstance(test_year, Iterable):
            test_year = min(test_year)

        # PREVENT temporal leakage of information
        min_test_date = pd.to_datetime(f"{test_year}-01-01")
        max_train_date = min_test_date - Day(1)
        min_train_date = pd.to_datetime(dynamic_ds.time.min().values)
        dynamic_ds = dynamic_ds.sel(time=slice(min_train_date, max_train_date))

        # augment the static data with the variables from dynamic_ds
        original_vars = list(dynamic_ds.data_vars)
        if dynamic_ignore_vars is not None:
            vars_list = [v for v in original_vars if v not in dynamic_ignore_vars]
        else:
            vars_list = original_vars

        print(
            "Augmenting the static data with"
            f" {'global_means' if global_means else ''}"
            f" {'spatial_means' if spatial_means else ''}"
            f"for variables: {vars_list}"
        )

        # check they have the same coords and dtypes
        reference_coord = [c for c in static_ds.coords][0]
        assert reference_coord in list(dynamic_ds.coords), (
            f"Static: {list(static_ds.coords)}" f" Dynamic: {list(dynamic_ds.coords)}"
        )
        assert static_ds[reference_coord].dtype == dynamic_ds[reference_coord].dtype, (
            f"Static: {static_ds[reference_coord].dtype}"
            f" Dynamic: {dynamic_ds[reference_coord].dtype}"
        )

        # calculate ones same shape as the static data
        first_var = list(static_ds.data_vars)[0]
        ones = xr.ones_like(static_ds[first_var])

        # for each NON-IGNORED dynamic variable calculate global_mean / spatial_mean
        list_data_arrays: List[xr.DataArray] = []
        for var in vars_list:
            if global_means:
                # GLOBAL mean
                global_mean_values = dynamic_ds[var].mean()
                global_mean_da = (global_mean_values * ones).rename(
                    f"{var}_global_mean"
                )
                list_data_arrays.append(global_mean_da)
            if spatial_means:
                # spatial mean
                spatial_mean_values = dynamic_ds[var].mean(dim="time")
                spatial_mean_da = (spatial_mean_values * ones).rename(
                    f"{var}_spatial_mean"
                )
                list_data_arrays.append(spatial_mean_da)

        if list_data_arrays != []:
            # join these new calculated variables into the original
            ds = xr.combine_by_coords(list_data_arrays)
            static_ds = static_ds.merge(ds)

        return static_ds

    def create_legend_df(ds: xr.Dataset, variable: str):
        """Create a UNIQUE key-value mapping describing the one hot encoding"""
        mapping = dict(enumerate(set(ds[variable].values)))
        legend = (
            pd.DataFrame(
                {
                    "value": [f"{variable}_{v}" for v in mapping.values()],
                    "raw_value": [v for v in mapping.values()],
                    "variable_name": [variable for _ in range(len(mapping.values()))],
                },
                index=[k for k in mapping.keys()],
            )
            .reset_index()
            .rename(columns={"index": "key"})
        )
        return legend

    def _one_hot_encode(da: xr.DataArray, legend: pd.DataFrame):
        """create a boolean DataArray for that value of the feature in `da`"""
        assert all(
            np.isin(["key", "value", "variable_name"], legend.columns)
        ), f"require key, value and variable_name in legend.columns. Got: {legend.columns}"
        print(f"Calculating OHE features for {legend.variable_name.unique()[0]}")

        list_of_hot_encoded_das = []
        for idx, row in legend.iterrows():
            key, value = row.key, row.value
            # convert to 0, 1 for that value
            ones = xr.ones_like(da)
            ohe_da = ones.where(da == value, 0).clip(min=0, max=1)
            ohe_da = ohe_da.rename(f"{value}_one_hot")

            list_of_hot_encoded_das.append(ohe_da)

        ds = xr.combine_by_coords(list_of_hot_encoded_das)
        return ds

    def one_hot_encode_vars(
        self, ds: xr.Dataset, variables_for_ohe: List[str], static: bool = True
    ) -> xr.Dataset:
        """OHE each variable of the correct type"""
        warnings.warn("This function `one_hot_encode_vars` can take a while to run")
        if static is False:
            assert (
                False
            ), "Expecting this function to work with static data only for the moment"
        # iterate over each variable for one hot encoding
        for variable in variables_for_ohe:
            da = ds[variable]
            legend = create_legend_df(ds, variable)
            ohe_ds = _one_hot_encode(da, legend)

            # merge back into the original Dataset
            ds = ds.merge(ohe_ds)

            # save legend to csv
            if static:
                legend.to_csv(self.static_output_folder / f"{variable}_legend.csv")
            else:
                legend.to_csv(self.output_folder / f"{variable}_legend.csv")

        return ds
