from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd
from pandas import Timestamp
from random import shuffle
from pathlib import Path
import pickle
import torch
import xarray as xr

from typing import cast, Dict, Optional, Union, List, Tuple


@dataclass
class TrainData:
    historical: Union[np.ndarray, torch.Tensor]
    current: Union[np.ndarray, torch.Tensor, None]
    pred_months: Union[np.ndarray, torch.Tensor]
    # latlons are repeated here so they can be tensor-ized and
    # normalized
    latlons: Union[np.ndarray, torch.Tensor]
    yearly_aggs: Union[np.ndarray, torch.Tensor]
    static: Union[np.ndarray, torch.Tensor, None]


@dataclass
class ModelArrays:
    x: TrainData
    y: Union[np.ndarray, torch.Tensor]
    x_vars: List[str]
    y_var: str
    latlons: Optional[np.ndarray] = None
    target_time: Optional[Timestamp] = None


# The dict below maps the indices of the arrays returned
# by the training iterator to their name. It should be updated
# if new inputs are added
idx_to_input = {
    0: 'historical',
    1: 'pred_months',
    2: 'latlons',
    3: 'current',
    4: 'yearly_aggs',
    5: 'static'
}


def train_val_mask(mask_len: int, val_ratio: float = 0.3) -> Tuple[List[bool], List[bool]]:
    """Makes a training and validation mask which can be passed to the dataloader
    Arguments
    ----------
    mask_len: int
        The length of the mask to be created
    val_ratio: float = 0.3
        The ratio of instances which should be True for the val mask and False for the train
        mask
    Returns
    ----------
    The train mask and the val mask, both as lists
    """
    assert val_ratio < 1, f'Val ratio must be smaller than 1'
    train_mask = np.random.rand(mask_len) < 1 - val_ratio
    val_mask = ~train_mask

    return train_mask.tolist(), val_mask.tolist()


class DataLoader:
    """Dataloader; lazily load the training and test data
    Attributes:
    ----------
    data_path: Path = Path('data')
        Location of the data folder
    batch_file_size: int = 1
        The number of files to load at a time
    mode: str {'test', 'train'} = 'train'
        Whether to load testing or training data. This also affects the way the data is
        returned; for train, it is a concatenated array, but for test it is a dict with dates
        so that the netcdf file can easily be reconstructed
    shuffle_data: bool = True
        Whether or not to shuffle data
    clear_nans: bool = True
        Whether to remove nan values
    experiment: str = 'one_month_forecast'
        the name of the experiment to run. Defaults to one_month_forecast
        (train on only historical data and predict one month ahead)
    normalize: bool = True
        Whether to normalize the data. This assumes a normalizing_dict.pkl was saved by the
        engineer
    mask: Optional[List[bool]] = None
        If not None, this list will be used to mask the input files. Useful for creating a train
        and validation set
    pred_months: Optional[List[int]] = None
        The months the model should predict. If None, all months are predicted
    to_tensor: bool = False
        Whether to turn the np.ndarrays into torch.Tensors
    surrounding_pixels: Optional[int] = None
        How many surrounding pixels to add to the input data. e.g. if the input is 1, then in
        addition to the pixels on the prediction point, the neighbouring (spatial) pixels will
        be included too, up to a distance of one pixel away
    ignore_vars: Optional[List[str]] = None
        A list of variables to ignore. If None, all variables in the data_path will be included
    monthly_aggs: bool = True
        Whether to include the monthly aggregates (mean and std across all spatial values) for
        the input variables. These will be additional dimensions to the historical
        (and optionally current) arrays
    static: bool = True
        Whether to include static data
    """
    def __init__(self, data_path: Path = Path('data'), batch_file_size: int = 1,
                 mode: str = 'train', shuffle_data: bool = True,
                 clear_nans: bool = True, normalize: bool = True,
                 experiment: str = 'one_month_forecast',
                 mask: Optional[List[bool]] = None,
                 pred_months: Optional[List[int]] = None,
                 to_tensor: bool = False,
                 surrounding_pixels: Optional[int] = None,
                 ignore_vars: Optional[List[str]] = None,
                 monthly_aggs: bool = True,
                 static: bool = False,
                 device: str = 'cpu') -> None:

        self.batch_file_size = batch_file_size
        self.mode = mode
        self.shuffle = shuffle_data
        self.clear_nans = clear_nans
        self.experiment = experiment
        self.data_files = self._load_datasets(
            data_path=data_path, mode=mode, shuffle_data=shuffle_data,
            experiment=experiment, mask=mask, pred_months=pred_months
        )

        self.normalizing_dict = None
        if normalize:
            with (
                data_path / f'features/{experiment}/normalizing_dict.pkl'
            ).open('rb') as f:
                self.normalizing_dict = pickle.load(f)

        self.surrounding_pixels = surrounding_pixels
        self.monthly_aggs = monthly_aggs
        self.to_tensor = to_tensor
        self.ignore_vars = ignore_vars

        self.static = None
        self.static_normalizing_dict = None

        if static:
            static_data_path = data_path / 'features/static/data.nc'
            self.static = xr.open_dataset(static_data_path)

            if normalize:
                static_normalizer_path = data_path / 'features/static/normalizing_dict.pkl'
                with static_normalizer_path.open('rb') as f:
                    self.static_normalizing_dict = pickle.load(f)
        self.device = torch.device(device)

    def __iter__(self):
        if self.mode == 'train':
            return _TrainIter(self)
        else:
            return _TestIter(self)

    def __len__(self) -> int:
        return len(self.data_files) // self.batch_file_size

    @staticmethod
    def _load_datasets(data_path: Path, mode: str,
                       shuffle_data: bool, experiment: str,
                       mask: Optional[List[bool]] = None,
                       pred_months: Optional[List[int]] = None) -> List[Path]:

        data_folder = data_path / f'features/{experiment}/{mode}'
        output_paths: List[Path] = []

        for subtrain in data_folder.iterdir():
            if (subtrain / 'x.nc').exists() and (subtrain / 'y.nc').exists():
                if pred_months is None:
                    output_paths.append(subtrain)
                else:
                    month = int(str(subtrain.parts[-1])[5:])
                    if month in pred_months:
                        output_paths.append(subtrain)

        if mask is not None:
            output_paths.sort()
            assert len(output_paths) == len(mask), \
                f'Output path and mask must be the same length!'
            output_paths = [o_p for o_p, include in zip(output_paths, mask) if include]
        if shuffle_data:
            shuffle(output_paths)
        return output_paths


class _BaseIter:
    """Base iterator
    """
    def __init__(self, loader: DataLoader) -> None:
        self.data_files = loader.data_files
        self.batch_file_size = loader.batch_file_size
        self.shuffle = loader.shuffle
        self.clear_nans = loader.clear_nans
        self.surrounding_pixels = loader.surrounding_pixels
        self.monthly_aggs = loader.monthly_aggs
        self.to_tensor = loader.to_tensor
        self.experiment = loader.experiment
        self.ignore_vars = loader.ignore_vars
        self.device = loader.device

        self.static = loader.static
        self.static_normalizing_dict = loader.static_normalizing_dict

        self.static_array: Optional[np.ndarray] = None

        if self.shuffle:
            # makes sure they are shuffled every epoch
            shuffle(self.data_files)

        self.normalizing_dict = loader.normalizing_dict
        self.normalizing_array: Optional[Dict[str, np.ndarray]] = None
        self.normalizing_array_ym: Optional[Dict[str, np.ndarray]] = None
        self.normalizing_array_static: Optional[Dict[str, np.ndarray]] = None

        self.idx = 0
        self.max_idx = len(loader.data_files)

    def __iter__(self):
        return self

    def calculate_static_normalizing_array(self, data_vars: List[str]) -> Dict[str, np.ndarray]:

        self.static_normalizing_dict = cast(Dict[str, Dict[str, float]],
                                            self.static_normalizing_dict)

        mean, std = [], []

        normalizing_dict_keys = self.static_normalizing_dict.keys()
        for var in data_vars:
            for norm_var in normalizing_dict_keys:
                if var == norm_var:
                    mean.append(self.static_normalizing_dict[norm_var]['mean'])
                    std.append(self.static_normalizing_dict[norm_var]['std'])
                    break

        normalizing_array = cast(Dict[str, np.ndarray], {
            'mean': np.asarray(mean),
            'std': np.asarray(std)
        })
        return normalizing_array

    def calculate_normalizing_array(self, data_vars: List[str]) -> Dict[str, np.ndarray]:
        # If we've made it here, normalizing_dict is definitely not None
        self.normalizing_dict = cast(Dict[str, Dict[str, float]], self.normalizing_dict)

        mean, std = [], []
        normalizing_dict_keys = self.normalizing_dict.keys()
        for var in data_vars:
            for norm_var in normalizing_dict_keys:
                if var.endswith(norm_var):
                    mean.append(self.normalizing_dict[norm_var]['mean'])
                    std.append(self.normalizing_dict[norm_var]['std'])
                    break

        mean_np, std_np = np.asarray(mean), np.asarray(std)

        normalizing_array = cast(Dict[str, np.ndarray], {
            'mean': mean_np,
            'std': std_np
        })
        return normalizing_array

    def _calculate_aggs(self, x: xr.Dataset) -> np.ndarray:
        yearly_mean = x.mean(dim=['time', 'lat', 'lon'])
        yearly_agg = yearly_mean.to_array().values

        if (self.normalizing_dict is not None) and (self.normalizing_array is None):
            self.normalizing_array_ym = self.calculate_normalizing_array(
                list(yearly_mean.data_vars))

        if self.normalizing_array_ym is not None:
            yearly_agg = (yearly_agg - self.normalizing_array_ym['mean']) / \
                self.normalizing_array_ym['std']
        return yearly_agg

    def _calculate_historical(self, x: xr.Dataset,
                              y: xr.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        x = self._add_extra_dims(x, self.surrounding_pixels, self.monthly_aggs)

        x_np, y_np = x.to_array().values, y.to_array().values

        # first, x
        x_np = x_np.reshape(x_np.shape[0], x_np.shape[1], x_np.shape[2] * x_np.shape[3])
        x_np = np.moveaxis(np.moveaxis(x_np, 0, 1), -1, 0)

        # then, y
        y_np = y_np.reshape(y_np.shape[0], y_np.shape[1], y_np.shape[2] * y_np.shape[3])
        y_np = np.moveaxis(y_np, -1, 0).reshape(-1, 1)

        if (self.normalizing_dict is not None) and (self.normalizing_array is None):
            self.normalizing_array = self.calculate_normalizing_array(
                list(x.data_vars))
            x_np = (x_np - self.normalizing_array['mean']) / (self.normalizing_array['std'])

        return x_np, y_np

    @staticmethod
    def _calculate_target_months(y: xr.Dataset, num_instances: int) -> np.ndarray:
        # then, the x month
        assert len(y.time) == 1, 'Expected y to only have 1 timestamp!' \
            f'Got {len(y.time)}'
        target_month = datetime.strptime(
            str(y.time.values[0])[:-3], '%Y-%m-%dT%H:%M:%S.%f'
        ).month
        x_months = np.array([target_month] * num_instances)

        return x_months

    def _calculate_latlons(self, x: xr.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        # then, latlons
        lons, lats = np.meshgrid(x.lon.values, x.lat.values)
        flat_lats, flat_lons = lats.reshape(-1, 1), lons.reshape(-1, 1)
        latlons = np.concatenate((flat_lats, flat_lons), axis=-1)
        train_latlons = np.concatenate((flat_lats, flat_lons), axis=-1)

        if self.normalizing_array is not None:
            train_latlons / [90, 180]

        return latlons, train_latlons

    def _calculate_static(self, num_instances: int) -> np.ndarray:
        if self.static_array is None:
            static_np = self.static.to_array().values  # type: ignore
            static_np = static_np.reshape(static_np.shape[0],
                                          static_np.shape[1] * static_np.shape[2])
            static_np = np.moveaxis(static_np, -1, 0)
            assert static_np.shape[0] == num_instances

            if self.static_normalizing_dict is not None:
                vars = list(self.static.data_vars)  # type: ignore
                self.static_normalizing_array = self.calculate_static_normalizing_array(vars)

                static_np = (static_np - self.static_normalizing_array['mean']) / \
                    self.static_normalizing_array['std']
            self.static_array = static_np
        return self.static_array

    def ds_folder_to_np(self,
                        folder: Path,
                        clear_nans: bool = True,
                        to_tensor: bool = False
                        ) -> ModelArrays:

        x, y = xr.open_dataset(folder / 'x.nc'), xr.open_dataset(folder / 'y.nc')
        assert len(list(y.data_vars)) == 1, f'Expect only 1 target variable! ' \
            f'Got {len(list(y.data_vars))}'
        if self.ignore_vars is not None:
            x = x.drop(self.ignore_vars)

        target_time = pd.to_datetime(y.time.values[0])
        yearly_agg = self._calculate_aggs(x)  # before to avoid aggs from surrounding pixels
        x_np, y_np = self._calculate_historical(x, y)
        x_months = self._calculate_target_months(y, x_np.shape[0])
        yearly_agg = np.vstack([yearly_agg] * x_np.shape[0])
        if self.static is not None:
            static_np = self._calculate_static(x_np.shape[0])
        else:
            static_np = None

        latlons, train_latlons = self._calculate_latlons(x)

        if self.experiment == 'nowcast':
            # if nowcast then we have a TrainData.current
            historical = x_np[:, :-1, :]  # all timesteps except the final
            current = self.get_current_array(  # only select NON-TARGET vars
                x=x, y=y, x_np=x_np
            )

            train_data = TrainData(
                current=current,
                historical=historical,
                pred_months=x_months,
                latlons=train_latlons,
                yearly_aggs=yearly_agg,
                static=static_np
            )

        else:
            train_data = TrainData(
                current=None,
                historical=x_np,
                pred_months=x_months,
                latlons=train_latlons,
                yearly_aggs=yearly_agg,
                static=static_np
            )

        assert y_np.shape[0] == x_np.shape[0], f'x and y data have a different ' \
            f'number of instances! x: {x_np.shape[0]}, y: {y_np.shape[0]}'

        if clear_nans:
            # remove nans if they are in the x or y data
            historical_nans, y_nans = np.isnan(train_data.historical), np.isnan(y_np)
            if train_data.static is not None:
                static_nans = np.isnan(train_data.static)
                static_nans_summed = static_nans.sum(axis=-1)
            else:
                static_nans_summed = np.zeros((y_nans.shape[0], ))

            historical_nans_summed = historical_nans.reshape(
                historical_nans.shape[0], historical_nans.shape[1] * historical_nans.shape[2]
            ).sum(axis=-1)
            y_nans_summed = y_nans.sum(axis=-1)

            notnan_indices = np.where((historical_nans_summed == 0) & (y_nans_summed == 0) &
                                      (static_nans_summed == 0))[0]

            if self.experiment == 'nowcast':
                current_nans = np.isnan(train_data.current)
                current_nans_summed = current_nans.sum(axis=-1)
                notnan_indices = np.where(
                    (
                        historical_nans_summed == 0
                    ) & (y_nans_summed == 0) & (
                        current_nans_summed == 0
                    ) & (static_nans_summed == 0)
                )[0]
                train_data.current = train_data.current[notnan_indices]  # type: ignore

            train_data.historical = train_data.historical[notnan_indices]
            train_data.pred_months = train_data.pred_months[notnan_indices]  # type: ignore
            train_data.latlons = train_data.latlons[notnan_indices]
            train_data.yearly_aggs = train_data.yearly_aggs[notnan_indices]
            if train_data.static is not None:
                train_data.static = train_data.static[notnan_indices]

            y_np = y_np[notnan_indices]

            latlons = latlons[notnan_indices]

        if to_tensor:
            train_data.historical = torch.from_numpy(train_data.historical).to(self.device).\
                float()
            train_data.pred_months = torch.from_numpy(train_data.pred_months).to(self.device).\
                float()
            train_data.latlons = torch.from_numpy(train_data.latlons).to(self.device).float()
            train_data.yearly_aggs = torch.from_numpy(train_data.yearly_aggs).to(self.device).\
                float()
            if train_data.static is not None:
                train_data.static = torch.from_numpy(train_data.static).to(self.device).float()
            y_np = torch.from_numpy(y_np).to(self.device).float()

            if self.experiment == 'nowcast':
                train_data.current = torch.from_numpy(train_data.current).to(self.device).\
                    float()
        return ModelArrays(x=train_data, y=y_np, x_vars=list(x.data_vars),
                           y_var=list(y.data_vars)[0], latlons=latlons,
                           target_time=target_time)

    @staticmethod
    def _add_extra_dims(x: xr.Dataset, surrounding_pixels: Optional[int],
                        monthly_agg: bool) -> xr.Dataset:
        original_vars = list(x.data_vars)

        if monthly_agg:
            # first, the means
            monthly_mean_values = x.mean(dim=['lat', 'lon'])
            mean_dataset = xr.ones_like(x) * monthly_mean_values

            for var in mean_dataset.data_vars:
                x[f'spatial_mean_{var}'] = mean_dataset[var]

        if surrounding_pixels is not None:
            lat_shifts = lon_shifts = range(-surrounding_pixels, surrounding_pixels + 1)
            for var in original_vars:
                for lat_shift in lat_shifts:
                    for lon_shift in lon_shifts:
                        if lat_shift == lon_shift == 0: continue
                        shifted_varname = f'lat_{lat_shift}_lon_{lon_shift}_{var}'
                        x[shifted_varname] = x[var].shift(lat=lat_shift, lon=lon_shift)
        return x

    @staticmethod
    def get_current_array(x: xr.Dataset, y: xr.Dataset, x_np: np.ndarray) -> np.ndarray:
        # get the target variable
        target_var = [y for y in y.data_vars][0]

        # get the target time and target_time index
        target_time = y.time
        x_datetimes = [pd.to_datetime(time) for time in x.time.values]
        y_datetime = pd.to_datetime(target_time.values[0])
        time_ix = [
            ix for ix, time in enumerate(x_datetimes)
            if time == y_datetime
        ][0]

        # get the X features and X feature indices
        relevant_indices = [
            idx for idx, feat in enumerate(x.data_vars)
            if not feat.endswith(target_var)
        ]

        # (latlon, time, data_var)
        current = x_np[:, time_ix, relevant_indices]

        assert len(current.shape) == 2, "Expected array: (lat*lon, time, dims)" \
            f"Got:{current.shape}"
        return current


class _TrainIter(_BaseIter):

    def __next__(self) -> Tuple[Tuple[Union[np.ndarray, torch.Tensor],
                                      ...],
                                Union[np.ndarray, torch.Tensor]]:

        if self.idx < self.max_idx:
            out_x, out_x_add, out_x_curr, out_x_latlon = [], [], [], []
            out_x_ym, out_x_static = [], []
            out_y = []

            cur_max_idx = min(self.idx + self.batch_file_size, self.max_idx)
            while self.idx < cur_max_idx:
                subfolder = self.data_files[self.idx]
                arrays = self.ds_folder_to_np(
                    subfolder, clear_nans=self.clear_nans,
                    to_tensor=False
                )
                if arrays.x.historical.shape[0] == 0:
                    print(f'{subfolder} returns no values. Skipping')

                    # remove the empty element from the list
                    self.data_files.pop(self.idx)
                    self.max_idx -= 1
                    self.idx -= 1  # we're going to add one later

                    cur_max_idx = min(cur_max_idx + 1, self.max_idx)

                out_x.append(arrays.x.historical)
                if arrays.x.current is not None:
                    out_x_curr.append(arrays.x.current)
                if arrays.x.static is not None:
                    out_x_static.append(arrays.x.static)
                out_x_add.append(arrays.x.pred_months)
                out_x_latlon.append(arrays.x.latlons)
                out_x_ym.append(arrays.x.yearly_aggs)
                out_y.append(arrays.y)
                self.idx += 1

            final_x = np.concatenate(out_x, axis=0)
            final_x_curr = np.concatenate(out_x_curr, axis=0) if out_x_curr != [] else None
            final_x_static = np.concatenate(out_x_static, axis=0) if out_x_static != [] else None
            final_x_add = np.concatenate(out_x_add, axis=0)
            final_x_latlon = np.concatenate(out_x_latlon, axis=0)
            final_x_ym = np.concatenate(out_x_ym, axis=0)
            final_y = np.concatenate(out_y, axis=0)

            if self.to_tensor:
                final_x = torch.from_numpy(final_x).to(self.device).float()
                final_x_add = torch.from_numpy(final_x_add).to(self.device).float()
                final_x_latlon = torch.from_numpy(final_x_latlon).to(self.device).float()
                final_x_ym = torch.from_numpy(final_x_ym).to(self.device).float()
                if final_x_curr is not None:
                    final_x_curr = torch.from_numpy(final_x_curr).to(self.device).float()
                if final_x_static is not None:
                    final_x_static = torch.from_numpy(final_x_static).to(self.device).float()
                final_y = torch.from_numpy(final_y).to(self.device).float()

            if final_x.shape[0] == 0:
                raise StopIteration()

            return (final_x, final_x_add, final_x_latlon, final_x_curr,
                    final_x_ym, final_x_static), final_y

        else:  # final_x_curr >= self.max_idx
            raise StopIteration()


class _TestIter(_BaseIter):

    def __next__(self) -> Dict[str, ModelArrays]:

        if self.idx < self.max_idx:
            out_dict = {}

            cur_max_idx = min(self.idx + self.batch_file_size, self.max_idx)
            while self.idx < cur_max_idx:
                subfolder = self.data_files[self.idx]
                arrays = self.ds_folder_to_np(subfolder, clear_nans=self.clear_nans,
                                              to_tensor=self.to_tensor)
                if arrays.x.historical.shape[0] == 0:
                    print(f'{subfolder} returns no values. Skipping')
                    # remove the empty element from the list
                    self.data_files.pop(self.idx)
                    self.max_idx -= 1
                    cur_max_idx = min(cur_max_idx + 1, self.max_idx)
                else:
                    out_dict[subfolder.parts[-1]] = arrays
                self.idx += 1

            if len(out_dict) == 0:
                raise StopIteration()
            return out_dict
        else:
            raise StopIteration()
