from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
from typing import Dict, List, Tuple, Union, Optional
import pickle
import tqdm
from torch.utils.data import Dataset
import torch
import h5py
import sys
from .runoff_utils import reshape_data


class CamelsCSV(Dataset):
    """Load data for individual basin to np.ndarrays to be used
    as input to the models

    Attrs:
        basin (str): [description]
        train_dates (List[int]): [description]
        data_dir (Path, optional): [description]. Defaults to Path("data").
        is_train (bool, optional): [description]. Defaults to True.
        target_var (str, optional): [description]. Defaults to "discharge_spec".
        x_variables (Optional[List[str]], optional): [description]. Defaults to ["precipitation", "peti"].
        static_variables (Optional[List[str]], optional): [description]. Defaults to None.
        ignore_static_vars (Optional[List[str]], optional): [description]. Defaults to None.
        seq_length (int, optional): [description]. Defaults to 365.
        with_static (bool, optional): [description]. Defaults to False.
        concat_static (bool, optional): [description]. Defaults to False.

    Created Attrs:
        gauge_ids (List[int]):
        target_std (float): the target_variable standard deviation, used for NSE calculation
        x (np.ndarray):  shape == (n_samples, seq_length, )
        y (np.ndarray):  shape == (n_samples, 1)
        num_samples (int): number of samples (timesteps)
    """

    def __init__(
        self,
        basin: str,
        train_dates: List[int],  # dates
        normalization_dict: Dict[str, float],
        data_dir: Path = Path("data"),
        is_train: bool = True,
        target_var: str = "discharge_spec",
        x_variables: Optional[List[str]] = ["precipitation", "peti"],
        static_variables: Optional[List[str]] = None,
        ignore_static_vars: Optional[List[str]] = None,
        seq_length: int = 365,
        with_static: bool = False,
        concat_static: bool = False,
    ):
        self.data_dir = data_dir
        self.normalization_dict = normalization_dict

        # initialise paths
        base_camels_dir = self.data_dir / "raw/CAMELS_GB_DATASET"
        self.attributes_dir = base_camels_dir / "Catchment_Attributes"
        self.timeseries_dir = base_camels_dir / "Catchment_Timeseries"
        self.shp_path = (
            base_camels_dir / "Catchment_Boundaries/CAMELS_GB_catchment_boundaries.shp"
        )
        assert self.attributes_dir.exists()
        assert self.timeseries_dir.exists()

        # data globals
        self.seq_length = seq_length
        self.target_var = target_var
        self.x_variables = x_variables
        self.static_variables = static_variables
        self.ignore_static_vars = ignore_static_vars

        # iteration params
        self.basin = basin
        self.is_train = is_train
        self.train_dates = np.sort(train_dates)
        assert all(
            [
                (isinstance(date, int) or (isinstance(int(date), int)))
                for date in train_dates
            ]
        ), "train_dates must be an array of integers (the years to be used)"

        # means and stds
        self.with_static = with_static
        self.concat_static = concat_static

        # get static data (preprocess if not preprocessed)
        # get all filepaths for csv files in CAMELS_GB_DATASET folder
        self.ts_csvs = [d for d in self.timeseries_dir.glob("*.csv")]
        self.gauge_ids = [
            int(d.name.split("ies_")[-1].split("_")[0])
            for d in self.timeseries_dir.glob("*.csv")
        ]

        # placeholder to store std of discharge, used for rescaling losses during training
        self.target_std = None

        # placeholder to store start and end date of entire period (incl warmup)
        self.period_start = None
        self.period_end = None
        self.attribute_names = None
        # array of nans for missing timesteps
        self.missing_timesteps: np.ndarray

        self.x, self.y = self._load_dynamic_data()

        if self.with_static:
            self.attributes = self._load_static_data()

        # NUM timesteps
        self.num_samples = self.x.shape[0]

        # get the valid dates for all of the timesteps found in the data
        self.date_range = self.get_date_range()

    def __len__(self):
        return self.num_samples

    def __getitem__(
        self, idx: int
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """iterate through the timesteps for each station
        returning X_dyn, X_stat, y.

        Args:
            idx (int): timestep number

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: X, y
            OR
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: X_dyn, X_stat y
        """
        if self.with_static:
            if self.concat_static:  # return concat(dynamic, static), target
                x = torch.cat(
                    [self.x[idx], self.attributes.repeat((self.seq_length, 1))], dim=-1
                )
                return x, self.y[idx]
            else:  # return dynamic, static, target
                return self.x[idx], self.attributes, self.y[idx]
        else:  # return only dynamic / target
            return self.x[idx], self.y[idx]

    def _get_one_basin_csv_file(self) -> Path:
        # get gauge_id
        gauge_ids = [
            d.name.split("ies_")[-1].split("_")[0]
            for d in self.timeseries_dir.glob("*.csv")
        ]
        bool_list = [str(id) == str(self.basin) for id in gauge_ids]
        assert (
            sum(bool_list) == 1
        ), f"Only expect to find one gauge id with id : {self.basin}"

        csv_file = np.array([d for d in self.timeseries_dir.glob("*.csv")])[bool_list][
            0
        ]

        return csv_file

    def _load_basin_dynamic_data(self):
        # load a single basin!
        csv_file = self._get_one_basin_csv_file()
        df = pd.read_csv(csv_file)

        # set datatypes
        df["date"] = df["date"].astype(np.dtype("datetime64[ns]"))
        df = df.set_index("date")
        df = df.astype("float")

        return df

    def _get_dynamic_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the dynamic X variables and the target variable (y)
        """
        basin_df = self._load_basin_dynamic_data()

        # get the features of interest
        if self.x_variables is not None:
            basin_df = basin_df[self.x_variables + [self.target_var]]

        # get the time series of interest
        basin_df = self._get_ts_of_interest(basin_df)

        self.period_start = basin_df.index[0]
        self.period_end = basin_df.index[-1]

        # load into numpy arrays (except target var)
        x = np.array(
            [
                basin_df[variable].values
                for variable in basin_df.columns
                if variable != self.target_var
            ]
        ).T

        # y should be shape (n_times, 1)
        y = np.array(basin_df[self.target_var].values).T
        y = y.reshape(-1, 1)

        return x, y

    def _remove_nans(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Remove nans from X and y

        Args:
            x (np.ndarray): the input data to the model
            y (np.ndarray): the target data for the model

        Returns:
            Tuple[np.ndarray, np.ndarray]: x, y
            np.ndarray: array of the nans indicating timestep missing
        """
        # any nans in y? (2D)
        y_nans = np.any(np.isnan(y), axis=1)
        # any nans in x? (3D)
        x_nans = np.any(np.any(np.isnan(x), axis=1), axis=1)

        all_nans = np.any([y_nans, x_nans], axis=0)
        self.missing_timesteps = all_nans
        total_nans = all_nans.sum()

        y = y[~all_nans]
        x = x[~all_nans]

        if total_nans > 0:
            print(f"{total_nans} NaNs removed in Basin: {self.basin}")

        return x, y

    def _load_dynamic_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self._get_dynamic_data()

        # normalize X data
        x = (x - self.normalization_dict["dynamic_means"]) / self.normalization_dict[
            "dynamic_stds"
        ]

        # create sequences of X -> y pairs
        x, y = reshape_data(x, y, self.seq_length)

        # delete nans
        x, y = self._remove_nans(x, y)

        if self.is_train:
            # store std of discharge BEFORE normalisation
            self.target_std = np.std(y)
            y = (y - self.normalization_dict["target_mean"]) / self.normalization_dict[
                "target_std"
            ]

        # convert to torch tensors
        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32))

        return x, y

    def _get_ts_of_interest(self, basin_df: pd.DataFrame) -> pd.DataFrame:
        """Create a new dataframe with only the datetimes that will be
        included in the X data.

        Args:
            basin_df (pd.DataFrame): dynamic data

        Returns:
            pd.DataFrame: return the data only for the timeseries of interest
        """
        start_date = pd.to_datetime(f"{self.train_dates[0]}-01-01") - pd.DateOffset(
            days=self.seq_length - 1
        )
        end_date = pd.to_datetime(f"{self.train_dates[-1]}-12-31")
        basin_df = basin_df[start_date:end_date]

        return basin_df

    def _create_static_data_xr(self) -> xr.Dataset:
        #  create the static data once
        attrs_csvs = [d for d in self.attributes_dir.glob("*.csv")]
        static_dfs = [pd.read_csv(d) for d in attrs_csvs]
        # join into one dataframe
        static_df = pd.concat(static_dfs, axis=1)

        # create xr object
        static_vars = [c for c in static_df.columns if c != "gauge_id"]
        dims = ["station_id"]
        coords = {"station_id": static_df.gauge_id.iloc[:, 0].values}

        static_ds = xr.Dataset(
            {
                variable_name: (dims, static_df[variable_name].values)
                for variable_name in static_vars
            },
            coords=coords,
        )

        # Fix the coordinates
        static_ds["station_id"] = static_ds["station_id"].astype(np.dtype("int64"))
        static_ds = static_ds.sortby("station_id")

        static_ds.to_netcdf(self.data_dir / "interim/static/data.nc")

        return static_ds

    def _load_static_data(self) -> torch.Tensor:
        """Load the static attributes (catchment attributes)
        as a

        Returns:
            torch.Tensor: Static data for this basin
        """
        static_df = self._get_static_data()

        # store attributes as PyTorch Tensor
        static_data = static_df.loc[static_df.index == self.basin].values

        return torch.from_numpy(static_data.astype(np.float32))

    def _get_static_data(self) -> pd.DataFrame:
        """Return the static data as an pd.DataFrame, read from the
        xr.Dataset (netcdf file).

        Returns:
            pd.DataFrame: Static catchment attributes (static over time)
        """
        # load from the xarray object
        if (self.data_dir / "interim/static/data.nc").exists():
            static_ds = xr.open_dataset(self.data_dir / "interim/static/data.nc")
        else:
            static_ds = self._create_static_data_xr()

        # drop vars from all studies
        if self.ignore_static_vars is not None:
            static_ds = static_ds.drop(
                [v for v in self.ignore_static_vars if v in static_ds.data_vars]
            )

        # keep only these vars
        if self.static_variables is not None:
            static_ds = static_ds[self.static_variables]

        static_df = static_ds.to_dataframe()

        # normalise the data
        static_df = (
            static_df - self.normalization_dict["static_means"]
        ) / self.normalization_dict["static_stds"]

        return static_df

    def get_date_range(self) -> pd.DatetimeIndex:
        # create date range
        date_range = pd.date_range(
            start=f"{self.train_dates[0]}-01-01",
            end=f"{self.train_dates[-1]}-12-31",
            freq="D",
        )

        # select the min/max date range
        period_start = self.period_start + pd.DateOffset(days=self.seq_length - 1)

        # create the date range (pd.Timestamps)
        date_range = date_range.to_series().loc[period_start : self.period_end].index

        # remove the missing dates
        date_range = date_range[~self.missing_timesteps]
        return date_range
