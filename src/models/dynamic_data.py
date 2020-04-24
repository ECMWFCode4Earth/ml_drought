from pandas.tseries.offsets import Day
import pandas as pd
import pickle
from collections import defaultdict
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime
from typing import DefaultDict, Dict, Tuple, Optional, Union, List, Any, cast
from random import shuffle
import torch
from collections import Iterable

from .data import DataLoader, ModelArrays, TrainData, _TrainIter, _TestIter
from src.utils import minus_timesteps

# ---------------------------------------------------------------------------
# DynamicDataLoader:
#  The DataLoader loads the data into memory from the engineered data.
#  It is initialised once at the start of each epoch.
#  It performs the following tasks:
#   - calculating the train/test split
#   - calculate the normalizing arrays (numpy) used by the iterators
#
# ---------------------------------------------------------------------------


class DynamicDataLoader(DataLoader):
    """
    New arguments:
    ---------------
    self.dynamic_ds: xr.Dataset
        load the data once into memory and then sample from it using the Iterators
    self.train_dynamic: xr.Dataset
    self.static_ds: xr.Dataset
        load the data once into memory and then sample from it using the Iterators

    self.legit_target_times: List[np.dtype('datetime64[ns]')]
    self.valid_train_times: List[np.dtype('datetime64[ns]')]
    self.valid_test_times: List[np.dtype('datetime64[ns]')]

    TODO: add in the logy as an argument to the dataloder not the engineer
    """

    dynamic: bool = True

    def __init__(
        self,
        target_var: str,
        test_years: Union[List[str], str],
        seq_length: int = 365,  # changed this default arg
        forecast_horizon: int = 1,
        data_path: Path = Path("data"),
        batch_file_size: int = 1,
        mode: str = "train",
        shuffle_data: bool = True,
        clear_nans: bool = True,
        normalize: bool = True,
        predict_delta: bool = False,
        experiment: str = "one_timestep_forecast",  # changed this default arg
        mask: Optional[List[bool]] = None,
        pred_months: Optional[List[int]] = None,
        to_tensor: bool = False,
        surrounding_pixels: Optional[int] = None,
        dynamic_ignore_vars: Optional[List[str]] = None,
        static_ignore_vars: Optional[List[str]] = None,
        timestep_aggs: bool = False,  # changed this default arg
        static: Optional[str] = "features",
        device: str = "cpu",
        spatial_mask: Optional[xr.DataArray] = None,
        normalize_y: bool = False,
        reducing_dims: Optional[List[str]] = None,
        calculate_latlons: bool = False,  # changed this default arg
        use_prev_y_var: bool = False,
        resolution: str = "D",
        logy: bool = False,
    ) -> None:
        self.data_path = data_path
        self.forecast_horizon = forecast_horizon
        self.target_var = target_var
        self.logy = logy

        self.dynamic_ignore_vars = dynamic_ignore_vars
        self.static_ignore_vars = static_ignore_vars
        # load engineered data into memory for sampling dynamically
        # TODO: split train/test sets HERE
        self.load_data(
            experiment=experiment,
            static=True if static is not None else False,
            test_years=test_years,
            mask=mask,
        )

        # init the super class
        super().__init__(
            data_path=data_path,
            batch_file_size=batch_file_size,
            mode=mode,
            shuffle_data=shuffle_data,
            clear_nans=clear_nans,
            normalize=normalize,
            predict_delta=predict_delta,
            experiment=experiment,
            mask=mask,
            seq_length=seq_length,
            pred_months=pred_months,
            to_tensor=to_tensor,
            surrounding_pixels=surrounding_pixels,
            ignore_vars=dynamic_ignore_vars,  # basically ignored
            timestep_aggs=timestep_aggs,
            static=static,
            device=device,
            spatial_mask=spatial_mask,
            normalize_y=normalize_y,
            reducing_dims=reducing_dims,
            calculate_latlons=calculate_latlons,
            use_prev_y_var=use_prev_y_var,
        )

        # get the legitimate target_timesteps
        assert self.seq_length is not None, "Must provide a sequence length. Try 365"
        self.resolution = resolution
        data_times = [pd.to_datetime(t) for t in self.dynamic_ds.time.values]

        # TODO: add mask here?
        self.legit_target_times = self.calculate_legitimate_target_times(
            seq_length=self.seq_length,
            data_times=data_times,
            resolution=self.resolution,
        )

        # TODO: calculate q_std for calculation of NSE
        # calculate the std of the target_var from the train_data period
        self.target_var_std = self.calculate_target_var_std()

    def __iter__(self):
        if self.mode == "train":
            return _TrainDynamicIter(self, mode="train")
        else:
            return _TestDynamicIter(self, mode="test")

    def __len__(self) -> int:
        if self.mode == "train":
            return len(self.valid_train_times) // self.batch_file_size
        else:
            return len(self.valid_test_times) // self.batch_file_size

    def get_train_test_times(
        self, test_years: Union[List[str], str], mask: Optional[List[bool]] = None
    ) -> Tuple[List[datetime], List[datetime]]:
        """Get a list of the test timestamps, train_timestamps"""
        min_test_date, max_train_date, _ = self.get_max_train_date(
            self.dynamic_ds, test_year=test_years
        )

        # use pandas .loc functionality with timeseries to get train_test periods
        ds_times = pd.DataFrame(index=pd.DatetimeIndex(self.dynamic_ds.time.values))
        valid_train_times = ds_times.loc[:max_train_date].index.values
        if mask:
            # get the train / validation data using mask
            valid_train_times = np.array(valid_train_times)[mask]
        valid_test_times = ds_times.loc[min_test_date:].index.values

        return valid_train_times, valid_test_times

    def calculate_target_var_std(self) -> np.ndarray:
        y_da = self.dynamic_ds.sel(time=self.valid_train_times)[self.target_var]
        return y_da.std(dim="time").values

    @staticmethod
    def get_max_train_date(
        ds, test_year: Union[str, List[str]]
    ) -> Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
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

    def log_target_variable(self, dynamic_ds: xr.Dataset, offset=0.001) -> xr.Dataset:
        dynamic_ds[self.target_var] = np.log(dynamic_ds[self.target_var] + offset)
        return dynamic_ds

    def unlog_target_variable(self, dynamic_ds: xr.Dataset, offset=0.001) -> xr.Dataset:
        dynamic_ds[self.target_var] = np.exp(dynamic_ds[self.target_var]) - offset
        return dynamic_ds

    def get_reducing_dims(self, reducing_dims: Optional[List[str]] = None) -> List[str]:
        return [c for c in self.dynamic_ds.coords if c != "time"]

    def load_data(
        self,
        experiment: str,
        static: bool,
        test_years: Union[List[str], str],
        mask: Optional[List[bool]] = None,
    ) -> None:
        """load static and dynamic data into memory for
        dynamic sampling!

        Get the train_ds vs. the whole dynamic_ds
        """
        data_folder = self.data_path / f"features/{experiment}"
        dynamic_path = data_folder / "data.nc"
        static_path = self.data_path / "features/static/data.nc"

        self.dynamic_ds: xr.Dataset = xr.open_dataset(dynamic_path)

        # split into TRAIN/TEST data
        self.valid_train_times: List[pd.Timestamp]
        self.valid_test_times: List[pd.Timestamp]
        self.valid_train_times, self.valid_test_times = self.get_train_test_times(
            test_years, mask=mask
        )
        self.train_dynamic = self.dynamic_ds.sel(time=self.valid_train_times)

        self.normalizing_dict: Dict = pickle.load(
            open(data_folder / "normalizing_dict.pkl", "rb")
        )

        # calculate the normalizing_array in the loader too
        # ignore_vars = only include the vars being used in X!
        if self.dynamic_ignore_vars is not None:
            data_vars: List[str] = [
                v
                for v in self.normalizing_dict.keys()
                if v not in self.dynamic_ignore_vars
            ]
        else:
            data_vars: List[str] = [v for v in self.normalizing_dict.keys()]

        data_vars = [v for v in data_vars if v != "target_var_original"]

        self.normalizing_array: Optional[
            Dict[str, np.ndarray]
        ] = self.calculate_normalizing_array(data_vars, static=False)

        if static:
            self.static_ds = xr.open_dataset(static_path)
            self.static_normalizing_dict: Dict = pickle.load(
                open(self.data_path / "features/static/normalizing_dict.pkl", "rb")
            )
            # calculate the normalizing_array in the loader too
            data_vars: List[str] = [
                v
                for v in self.static_normalizing_dict.keys()
                if v not in self.static_ignore_vars
            ]
            self.normalizing_array_static: Optional[
                Dict[str, np.ndarray]
            ] = self.calculate_normalizing_array(data_vars, static=True)
        else:
            assert False, "Should provide static data for this loader ..."
            self.static_ds = None
            self.static_normalizing_dict = None

    @staticmethod
    def calculate_legitimate_target_times(
        data_times: List[pd.Timestamp], seq_length: int, resolution: str = "D"
    ) -> List[pd.Timestamp]:
        """return a list of the target times that we have enough data for"""
        min_data_date = min(data_times)

        # if the minimum required data is greater than the minimum date
        # then it is a valid time
        legitimate_times = []
        for target_time in data_times:
            min_X_date = minus_timesteps(target_time, seq_length, resolution)
            if min_X_date > min_data_date:
                legitimate_times.append(target_time)

        return legitimate_times

    def calculate_normalizing_array(
        self, data_vars: List[str], static: bool
    ) -> Dict[str, np.ndarray]:
        """Use dictionary created by the engineers -> numpy arrays
        for normalisation of the numpy arrays for use
        at a later time (by the iterators)
        """
        # If we've made it here, normalizing_dict is definitely not None
        if static:
            self.static_normalizing_dict = cast(
                Dict[str, Dict[str, float]], self.static_normalizing_dict
            )
        else:
            self.normalizing_dict = cast(
                Dict[str, Dict[str, float]], self.normalizing_dict
            )

        mean, std = [], []
        normalizing_dict_keys = (
            self.static_normalizing_dict.keys()
            if static
            else self.normalizing_dict.keys()
        )
        for var in data_vars:
            for norm_var in normalizing_dict_keys:
                if var.endswith(norm_var):
                    mean_val = (
                        self.static_normalizing_dict[norm_var]["mean"]
                        if static
                        else self.normalizing_dict[norm_var]["mean"]
                    )
                    std_val = (
                        self.static_normalizing_dict[norm_var]["std"]
                        if static
                        else self.normalizing_dict[norm_var]["std"]
                    )
                    mean.append(mean_val)
                    std.append(std_val)
                    break

        mean_np, std_np = np.asarray(mean), np.asarray(std)

        normalizing_array = cast(
            Dict[str, np.ndarray], {"mean": mean_np, "std": std_np}
        )
        return normalizing_array


# ---------------------------------------------------------------------------
# ITERATORS:
#  Write the iterators which use the dynamic data and static data loaded
#  into memory by the DynamicDataLoader. They then sample from it, iterating
#  over the target_timesteps (either train_timesteps, or test_timesteps).
#  They performs tasks such as:
#   - normalizing the numpy arrays so the model sees normalized data
#   -
# ---------------------------------------------------------------------------


class _DynamicIter:
    """dynamically load data from the loader.dynamic_ds & loader.static_ds

    Create a new Iterator each EPOCH.
    """

    def __init__(self, loader: DynamicDataLoader, mode: str) -> None:
        # NEW
        self.target_var_std = loader.target_var_std
        self.target_var = loader.target_var
        self.legit_target_times = loader.legit_target_times
        self.seq_length = loader.seq_length

        self.dynamic_ignore_vars = loader.dynamic_ignore_vars
        self.static_ignore_vars = loader.static_ignore_vars
        self.dynamic_ds = loader.dynamic_ds
        self.static_ds = loader.static_ds
        self.shuffle = loader.shuffle

        self.idx = 0

        self.valid_train_times = loader.valid_train_times
        self.valid_test_times = loader.valid_test_times
        self.forecast_horizon = loader.forecast_horizon

        if mode == "train":
            self.target_times = loader.valid_train_times
            self.target_times = [tt for tt in self.target_times]
        elif mode == "test":
            self.target_times = loader.valid_test_times
            self.target_times = [tt for tt in self.target_times]
        else:
            assert False, "Mode must be one of train / test"

        self.max_idx = len(self.target_times)

        # CHANGED
        if self.shuffle:
            # makes sure they are shuffled every epoch
            shuffle(self.target_times)
        # load directly from the DataLoader
        self.normalizing_array: Optional[
            Dict[str, np.ndarray]
        ] = loader.normalizing_array
        self.normalizing_array_static: Optional[
            Dict[str, np.ndarray]
        ] = loader.normalizing_array_static

        # OLD
        self.batch_file_size = loader.batch_file_size
        self.shuffle = loader.shuffle
        self.clear_nans = loader.clear_nans
        self.to_tensor = loader.to_tensor
        self.experiment = loader.experiment
        self.device = loader.device
        self.normalize_y = loader.normalize_y

        self.normalizing_dict = loader.normalizing_dict
        self.normalizing_array_static = loader.normalizing_array_static
        self.static = loader.static

        # removed
        self.reducing_dims: List[str] = loader.reducing_dims
        # self.calculate_latlons = loader.calculate_latlons
        # self.use_prev_y_var: bool = loader.use_prev_y_var
        # self.static_array: Optional[np.ndarray] = None
        # self.surrounding_pixels = loader.surrounding_pixels
        # self.timestep_aggs = loader.timestep_aggs
        # self.predict_delta = loader.predict_delta
        # self.spatial_mask = loader.spatial_mask
        # self.normalize_y = loader.normalize_y
        # self.normalizing_array_ym: Optional[Dict[str, np.ndarray]] = None

    def __iter__(self):
        return self

    def build_loc_to_idx_mapping(
        self, x: xr.Dataset, notnan_indices: Optional[np.array] = None
    ) -> Dict:
        """ build a mapping from SPATIAL ID to the value
        (pixel, station_id, admin_unit) removing the nan indices

        Why? In order to track the spatial units so that we can
        rebuild the predictions in the `evaluate` function. This
        way it doesn't require (lat, lon) but also works for
        station_id // flattened data (regions etc.)
        """
        reducing_coords = [c for c in x.coords if c != "time"]
        if len(reducing_coords) == 1:
            # working with 1D data (pixel, area, stations etc.)
            # create a DUMMY latlon ([0], [idx])
            ids = np.arange(len(x[reducing_coords[0]].values))
            values = x[reducing_coords[0]].values
            if notnan_indices is not None:
                ids, values = ids[notnan_indices], values[notnan_indices]

            id_to_val_map = dict(zip(ids, values))
        else:
            assert False, "Haven't included other dimensions (only 1D or latlon)"

        return id_to_val_map

    def clear_train_data_nans(
        self, x: xr.Dataset, train_data: TrainData, y_np: np.array
    ) -> Tuple[np.ndarray, TrainData, np.ndarray, Dict[int, Any]]:
        """remove the nans from the x/y data (stored in a TrainData object)"""
        # remove nans if they are in the x or y data
        historical_nans, y_nans = np.isnan(train_data.historical), np.isnan(y_np)

        if train_data.static is not None:
            static_nans = np.isnan(train_data.static)
            static_nans_summed = static_nans.sum(axis=-1)
        else:
            static_nans_summed = np.zeros((y_nans.shape[0],))

        historical_nans_summed = historical_nans.reshape(
            historical_nans.shape[0],
            historical_nans.shape[1] * historical_nans.shape[2],
        ).sum(axis=-1)
        y_nans_summed = y_nans.sum(axis=-1)

        notnan_indices = np.where(
            (historical_nans_summed == 0)
            & (static_nans_summed == 0)
            & (y_nans_summed == 0)
        )[0]

        train_data.filter(notnan_indices)

        y_np = y_np[notnan_indices]

        # store the mapping from ID -> pixel/spatial code
        id_to_loc_map = self.build_loc_to_idx_mapping(x, notnan_indices=notnan_indices)

        return notnan_indices, train_data, y_np, id_to_loc_map

    def _calculate_static(self, num_instances: int) -> np.ndarray:
        """Create the static numpy array from the static_ds xr.Dataset object

        Arguments:
        ---------
        : num_instances: int
            the number of spatial instances / examples (e.g. stations or pixels)
        """
        static_np: np.ndarray = self.static_ds.to_array().values

        if len(static_np.shape) == 3:
            #  if 3 dimensions (vars, lat, lon)
            # collapse to 2 dimensions (vars, pixels)
            static_np = static_np.reshape(
                static_np.shape[0], static_np.shape[1] * static_np.shape[2]
            )
        static_np = np.moveaxis(static_np, -1, 0)
        assert static_np.shape[0] == num_instances

        if self.normalizing_array_static is not None:
            static_np = (
                (static_np - self.normalizing_array_static["mean"])
                / [s if s != 0 else 1e-10 for s in self.normalizing_array_static["std"]]
                # TODO: only use STD if non-zero!
            )

            self.static_array = static_np
        return self.static_array

    def _calculate_historical(
        self, x: xr.Dataset, y: xr.Dataset
    ) -> Tuple[np.ndarray, np.ndarray]:

        if "target_var_original" in list(x.data_vars):
            x = x.drop("target_var_original")
        x_np, y_np = x.to_array().values, y.to_array().values

        # first, x
        if len(x_np.shape) == 4:
            #  if 4 dimensions (time, vars, lat, lon)
            # then collapse to 3 d (time, vars, latlon)
            x_np = x_np.reshape(
                x_np.shape[0], x_np.shape[1], x_np.shape[2] * x_np.shape[3]
            )
        x_np = np.moveaxis(np.moveaxis(x_np, 0, 1), -1, 0)

        # then, y
        if len(x_np.shape) == 4:  #  if 4 dimensions (time, vars, lat, lon)
            y_np = y_np.reshape(
                y_np.shape[0], y_np.shape[1], y_np.shape[2] * y_np.shape[3]
            )
        y_np = np.moveaxis(y_np, -1, 0).reshape(-1, 1)

        if self.normalizing_array is not None:
            x_np = (x_np - self.normalizing_array["mean"]) / (
                self.normalizing_array["std"]
            )
        else:
            assert (
                False
            ), "normalizing_dict should be provided by the Dynamic DataLoader/Engineer"

        # normalizing_dict will not be None
        # NORMALIZE y by default
        if self.normalize_y:
            y_var = list(y.data_vars)[0]
            y_np = (
                (y_np - self.normalizing_dict[y_var]["mean"])
                / self.normalizing_dict[y_var]["std"]  # type: ignore
            )

        return x_np, y_np

    def get_sample_from_dynamic_data(
        self,
        target_var: str,
        target_time: np.datetime64,
        seq_length: int,
        dynamic_ignore_vars: Optional[List[str]] = None,
        forecast_horizon: int = 1,
        train: bool = True,
        resolution: str = "D",
    ) -> Tuple[Union[None, Tuple[xr.Dataset, xr.Dataset]], pd.Timestamp]:
        """Get the X, y pair from the dynamic data for a given
        target_timestep.

        Arguments:
        ---------
        target_var: str
        target_time: np.datetime64
        seq_length: int
            How long is the sequence length for the calculation of the
        forecast_horizon: int = 1
        min_ds_date: pd.Timestamp
            If we have a min_X_date smaller than min_ds_date
            then we don't have sufficient training examples to
            calculate the
        forecast_horizon: int = 1
            number of timesteps into the future to forecast
        train: bool = True
        resolution: str = "D"
        """
        min_ds_date = pd.to_datetime(self.dynamic_ds.time.min().values)

        # convert to pandas datetime (easier to work with)
        target_time = pd.to_datetime(target_time)

        # min/max date for X data
        # forecast_horizon: how many days gap between X -> y
        max_X_date = minus_timesteps(target_time, forecast_horizon, resolution)
        min_X_date = minus_timesteps(target_time, seq_length, resolution)

        # check whether enough data
        if min_X_date < min_ds_date:
            print(f"Not enough input timesteps for {target_time}")
            return None, target_time

        # get the X, y pairs
        X_dataset = self.dynamic_ds.sel(time=slice(min_X_date, max_X_date))
        y_dataset = self.dynamic_ds[[target_var]].sel(time=target_time)

        # ignore vars (especially if been transformed e.g. log-tranform discharge_spec)
        if dynamic_ignore_vars is None:
            dynamic_ignore_vars = ["target_var_original"]
        else:
            dynamic_ignore_vars += ["target_var_original"]
            dynamic_ignore_vars = [
                v
                for v in dynamic_ignore_vars
                if v in [var_ for var_ in X_dataset.data_vars]
            ]
            X_dataset = X_dataset.drop(dynamic_ignore_vars)

        return (X_dataset, y_dataset), target_time

    @staticmethod
    def _calculate_target_months(y: xr.Dataset, num_instances: int) -> np.ndarray:
        # then, the x month
        try:
            assert len(y.time) == 1, (
                "Expected y to only have 1 timestamp!" f"Got {len(y.time)}"
            )
            time_value = y.time.values[0]
        except TypeError:
            assert y.time.shape == (), "Expected y to only have 1 timestamp!"
            time_value = y.time.values

        target_month = datetime.strptime(
            str(time_value)[:-3], "%Y-%m-%dT%H:%M:%S.%f"
        ).month
        x_months = np.array([target_month] * num_instances)

        return x_months

    def ds_sample_to_np(
        self,
        xy_sample: Tuple[xr.Dataset, xr.Dataset],
        target_time: pd.Timestamp,
        clear_nans: bool = True,
        to_tensor: bool = False,
        reducing_dims: List[str] = ["lat", "lon"],
        target_var_std: Optional[np.ndarray] = None,
    ) -> ModelArrays:
        """Convert the xr.Dataset objects in xy_sample into ModelArrays
        for passing to the models. This function works dynamically rather
        than reading from files (as in `ds_folder_to_np`)

        NOTE:
        - Removed the nowcasting ability
        - Removed the predict delta
        - Removed the spatial_mask
        - Removed the calculation of aggs (done before in the static data)
        """
        # vars are already ignored in `get_sample_from_dynamic_data`
        x, y = xy_sample

        assert len(list(y.data_vars)) == 1, (
            f"Expect only 1 target variable! " f"Got {len(list(y.data_vars))}"
        )

        # get the x_datetimes as list of pd.Timestamps
        x_datetimes = [pd.to_datetime(time) for time in x.time.values]

        # 1. create the numpy arrays
        ## normalize values in these functions
        x_np, y_np = self._calculate_historical(x, y)
        x_months = self._calculate_target_months(y, x_np.shape[0])

        ## static data
        # TODO: load the static data from the static_ds.nc
        if self.static is not None:
            static_np = self._calculate_static(x_np.shape[0])
        else:
            static_np = None

        # 2. Build the X data object
        train_data = TrainData(
            historical=x_np,
            pred_month=x_months,
            yearly_aggs=None,
            current=None,
            latlons=None,
            static=static_np,
            prev_y_var=None,
            target_var_std=target_var_std,
        )

        # 3. clear the nans
        if clear_nans:
            notnan_indices, train_data, y_np, id_to_loc_map, target_var_std_np = self.clear_train_data_nans(
                x, train_data, y_np
            )
        else:
            id_to_loc_map = self.build_loc_to_idx_mapping(x, notnan_indices=None)

        # train_data.target_var_std = target_var_std_np

        # 4. Create the ModelArrays (X-y pairs)
        y_var = list(y.data_vars)[0]
        model_arrays = ModelArrays(
            x=train_data,
            y=y_np,
            x_vars=list(x.data_vars),
            y_var=y_var,
            latlons=None,
            target_time=target_time,
            historical_times=x_datetimes,
            id_to_loc_map=id_to_loc_map,
        )

        if to_tensor:
            model_arrays.to_tensor(self.device)

        return model_arrays

    def deal_with_no_values(self, target_time: pd.Timestamp, cur_max_idx: int) -> int:
        print(f"{pd.to_datetime(target_time)} returns no values. Skipping")

        # remove the empty element from the list
        self.target_times.pop(self.idx)
        self.max_idx -= 1
        cur_max_idx = min(cur_max_idx + 1, self.max_idx)

        return cur_max_idx


class _TrainDynamicIter(_DynamicIter):
    """
    NOTE: only difference is we are iterating over
    self.target_times instead of the data_paths
    (List[pd.Timestamp] -> List[Path]
    """

    mode = "train"

    def __next__(
        self,
    ) -> Tuple[
        Tuple[Union[np.ndarray, torch.Tensor], ...], Union[np.ndarray, torch.Tensor]
    ]:
        # self.max_idx = len(self.target_times)

        # TODO: gabi why do we have these global_modelarrays?
        global_modelarrays: Optional[ModelArrays] = None
        cur_max_idx = min(self.idx + self.batch_file_size, self.max_idx)

        # iterate over each timestamp in the training data
        while self.idx < cur_max_idx:
            # iterate over the X-y pairs by selecting the target_time
            # dynamically from the self.dynamic_ds
            target_time = self.target_times[self.idx]

            xy_sample, _ = self.get_sample_from_dynamic_data(
                target_time=target_time,
                forecast_horizon=self.forecast_horizon,
                target_var=self.target_var,
                seq_length=self.seq_length,
                dynamic_ignore_vars=self.dynamic_ignore_vars,
            )
            if xy_sample is None:
                cur_max_idx = self.deal_with_no_values(
                    target_time=target_time, cur_max_idx=cur_max_idx
                )
            else:
                arrays = self.ds_sample_to_np(
                    xy_sample=xy_sample,
                    target_time=target_time,
                    clear_nans=self.clear_nans,
                    to_tensor=self.to_tensor,
                    reducing_dims=self.reducing_dims,
                    target_var_std=self.target_var_std,
                )

                # If there are no values!
                if arrays.x.historical.shape[0] == 0:
                    cur_max_idx = self.deal_with_no_values(
                        target_time=target_time, cur_max_idx=cur_max_idx
                    )
                    # print(f"{pd.to_datetime(target_time)} returns no values. Skipping")

                    # # remove the empty element from the list
                    # self.data_files.pop(self.idx)
                    # self.max_idx -= 1
                    # cur_max_idx = min(cur_max_idx + 1, self.max_idx)

                if global_modelarrays is None:
                    global_modelarrays = arrays
                else:
                    global_modelarrays.concatenate(arrays)
            self.idx += 1

            if global_modelarrays is not None:
                # convert to Tensor object (pytorch)
                if self.to_tensor:
                    global_modelarrays.to_tensor(self.device)

                # return the modelarrays X, y pairs
                return (
                    (
                        global_modelarrays.x.historical,
                        global_modelarrays.x.pred_month,
                        global_modelarrays.x.latlons,
                        global_modelarrays.x.current,
                        global_modelarrays.x.yearly_aggs,
                        global_modelarrays.x.static,
                        global_modelarrays.x.prev_y_var,
                    ),
                    global_modelarrays.y,
                )
            else:
                raise StopIteration()

        else:  # final_x_curr >= self.max_idx
            raise StopIteration()


class _TestDynamicIter(_DynamicIter):
    mode = "test"
    # max index is the final test time drawn from the DataLoader
    # max_idx = len(self.target_times)
    def make_timestamp(self, target_time: datetime) -> str:
        """TODO:
            - hardcoded for the DAILY data (deal with other resolution data e.g. months)
        """
        dt = pd.to_datetime(target_time)
        timestamp = f"{dt.year}-{dt.month}-{dt.day}"
        return timestamp

    def __next__(self) -> Dict[str, ModelArrays]:
        # self.max_idx = len(self.target_times)
        if self.idx < self.max_idx:
            out_dict = {}

            cur_max_idx = min(self.idx + self.batch_file_size, self.max_idx)
            while self.idx < cur_max_idx:

                # iterate over the X-y pairs by selecting the target_time
                # dynamically from the self.dynamic_ds
                target_time = self.target_times[self.idx]

                xy_sample, _ = self.get_sample_from_dynamic_data(
                    target_time=target_time,
                    forecast_horizon=self.forecast_horizon,
                    target_var=self.target_var,
                    seq_length=self.seq_length,
                    dynamic_ignore_vars=self.dynamic_ignore_vars,
                )

                arrays = self.ds_sample_to_np(
                    xy_sample=xy_sample,
                    target_time=target_time,
                    clear_nans=self.clear_nans,
                    to_tensor=self.to_tensor,
                    reducing_dims=self.reducing_dims,
                )

                # If there are no values!
                if arrays.x.historical.shape[0] == 0:
                    cur_max_idx = self.deal_with_no_values(
                        target_time=target_time, cur_max_idx=cur_max_idx
                    )
                else:
                    timestamp = self.make_timestamp(target_time)
                    out_dict[timestamp] = arrays
                self.idx += 1

            # assert that it's not empty
            if len(out_dict) == 0:
                raise StopIteration()
            return out_dict

        else:
            # end of sequence!
            raise StopIteration()
