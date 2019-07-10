from dataclasses import dataclass
from datetime import datetime
import numpy as np
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
    pred_months: Union[np.ndarray, torch.Tensor, None]


@dataclass
class ModelArrays:
    x: TrainData
    y: Union[np.ndarray, torch.Tensor]
    x_vars: List[str]
    y_var: str
    latlons: Optional[np.ndarray] = None


def train_val_mask(mask_len: int, val_ratio: float = 0.3) -> Tuple[List[bool], List[bool]]:
    """Makes a trainining and validation mask which can be passed to the dataloader

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
    normalize: bool = True
        Whether to normalize the data. This assumes a normalizing_dict.pkl was saved by the
        engineer
    mask: Optional[List[bool]] = None
        If not None, this list will be used to mask the input files. Useful for creating a train
        and validation set
    to_tensor: bool = False
        Whether to turn the np.ndarrays into torch.Tensors
    experiement: str = 'one_month_forecast'
        the name of the experiment to run. Defaults to one_month_forecast
        (train on only historical data and predict one month ahead)

    Note:
    the _load_datasets() function requires an `experiment` string defining
    which experiment is to be run. The string must be the same as the
    name of the experiment when creating the `x.npy` and `y.npy` `train`/`test`
    splits in the `engineer` classes.
    """
    def __init__(self, data_path: Path = Path('data'), batch_file_size: int = 1,
                 mode: str = 'train', shuffle_data: bool = True,
                 clear_nans: bool = True, normalize: bool = True,
                 experiment: str = 'one_month_forecast',
                 mask: Optional[List[bool]] = None,
                 to_tensor: bool = False) -> None:

        self.batch_file_size = batch_file_size
        self.mode = mode
        self.shuffle = shuffle_data
        self.clear_nans = clear_nans
        self.experiment = experiment
        self.data_files = self._load_datasets(
            data_path=data_path, mode=mode, shuffle_data=shuffle_data,
            experiment=experiment, mask=mask
        )

        self.normalizing_dict = None
        if normalize:
            with (
                data_path / f'features/{experiment}/normalizing_dict.pkl'
            ).open('rb') as f:
                self.normalizing_dict = pickle.load(f)

        self.to_tensor = to_tensor

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
                       mask: Optional[List[bool]] = None) -> List[Path]:

        data_folder = data_path / f'features/{experiment}/{mode}'
        output_paths: List[Path] = []

        for subtrain in data_folder.iterdir():
            if (subtrain / 'x.nc').exists() and (subtrain / 'y.nc').exists():
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
        self.to_tensor = loader.to_tensor
        self.experiment = loader.experiment

        if self.shuffle:
            # makes sure they are shuffled every epoch
            shuffle(self.data_files)

        self.normalizing_dict = loader.normalizing_dict
        self.historical_normalizing_array: Optional[Dict[str, np.ndarray]] = None
        self.current_normalizing_array: Optional[Dict[str, np.ndarray]] = None

        self.idx = 0
        self.max_idx = len(loader.data_files)

    def __iter__(self):
        return self

    def calculate_normalizing_array(self, data_vars: List[str], target_var: str):
        # If we've made it here, normalizing_dict is definitely not None
        self.normalizing_dict = cast(Dict[str, Dict[str, np.ndarray]], self.normalizing_dict)

        if self.experiment == 'nowcast':
            historical_mean, historical_std = [], []
            current_mean, current_std = [], []
            for var in data_vars:
                # historical_mean = all times up to current time
                historical_mean.append(self.normalizing_dict[var]['mean'][:-1])
                historical_std.append(self.normalizing_dict[var]['std'][:-1])

                # current_mean = the current times for all EXCEPT for target_var
                if var != target_var:
                    current_mean.append(self.normalizing_dict[var]['mean'][-1])
                    current_std.append(self.normalizing_dict[var]['std'][-1])

            self.current_normalizing_array = cast(Dict[str, np.ndarray], {
                # swapaxes so that its [timesteps, features], not [features, timesteps]
                'mean': np.vstack(current_mean).swapaxes(0, 1),
                'std': np.vstack(current_std).swapaxes(0, 1)
            })

            self.historical_normalizing_array = cast(Dict[str, np.ndarray], {
                # swapaxes so that its [timesteps, features], not [features, timesteps]
                'mean': np.vstack(historical_mean).swapaxes(0, 1),
                'std': np.vstack(historical_std).swapaxes(0, 1)
            })

            assert self.current_normalizing_array['mean'].shape[0] == 1, f"Current timestep can only be one for `self.current_normalizing_array`. Currently\
            {self.current_normalizing_array['mean'].shape}"

            expected = self.historical_normalizing_array['mean'].shape[-1]
            got = self.current_normalizing_array['mean'].shape[-1]
            assert got == (expected - 1), f"There should be one less variable in \
            `self.current_normalizing_array` compared to `self.current_normalizing_array`\
            historical: {self.historical_normalizing_array['mean'].shape}\
            current: {self.current_normalizing_array['mean'].shape}"

        else:
            mean, std = [], []
            for var in data_vars:
                mean.append(self.normalizing_dict[var]['mean'])
                std.append(self.normalizing_dict[var]['std'])

            self.current_normalizing_array = None
            self.historical_normalizing_array = cast(Dict[str, np.ndarray], {
                # swapaxes so that its [timesteps, features], not [features, timesteps]
                'mean': np.vstack(mean).swapaxes(0, 1),
                'std': np.vstack(std).swapaxes(0, 1)
            })

    def ds_folder_to_np(self,
                        folder: Path,
                        clear_nans: bool = True,
                        return_latlons: bool = False,
                        to_tensor: bool = False
                        ) -> ModelArrays:

        x, y = xr.open_dataset(folder / 'x.nc'), xr.open_dataset(folder / 'y.nc')
        assert len(list(y.data_vars)) == 1, f'Expect only 1 target variable! ' \
            f'Got {len(list(y.data_vars))}'

        coords = [c for c in y.coords]
        target_var = [y for y in y.variables if y not in coords][0]

        x_np, y_np = x.to_array().values, y.to_array().values

        if (self.normalizing_dict is not None) and (self.historical_normalizing_array is None):
            self.calculate_normalizing_array(list(x.data_vars), target_var=target_var)

        # first, x
        x_np = x_np.reshape(x_np.shape[0], x_np.shape[1], x_np.shape[2] * x_np.shape[3])
        x_np = np.moveaxis(np.moveaxis(x_np, 0, 1), -1, 0)

        # then, the x month
        assert len(y.time) == 1, 'Expected y to only have 1 timestamp!'\
            f'Got {len(y.time)}'
        target_month = datetime.strptime(
            str(y.time.values[0])[:-3], '%Y-%m-%dT%H:%M:%S.%f'
        ).month
        x_months = np.array([target_month] * x_np.shape[0])

        # then, y
        y_np = y_np.reshape(y_np.shape[0], y_np.shape[1], y_np.shape[2] * y_np.shape[3])
        y_np = np.moveaxis(y_np, -1, 0).reshape(-1, 1)

        if self.experiment == 'nowcast':
            # if nowcast then we have a TrainData.current
            historical = x_np[:, :-1, :]  # all timesteps except the final
            current = x_np[:, -1, :-1]  # final axis is all nan (target var)

            train_data = TrainData(
                current=current,
                historical=historical,
                pred_months=x_months  # @GABI do we want to include this ?
            )
        else:
            train_data = TrainData(
                current=None,
                historical=x_np,
                pred_months=x_months
            )

        if self.historical_normalizing_array is not None:
            if self.experiment == 'nowcast':
                # normalize the historical X data (all vars including target variable)
                train_data.historical = (
                    train_data.historical - self.historical_normalizing_array['mean']
                ) / (self.historical_normalizing_array['std'])

                # normalize the current X data (no target variable)
                if self.current_normalizing_array is not None:
                    train_data.current = (
                        train_data.current - self.current_normalizing_array['mean']
                    ) / (self.current_normalizing_array['std'])
            else:
                # only historical variables for non-nowcast experiments
                train_data.historical = (
                    train_data.historical - self.historical_normalizing_array['mean']
                ) / (self.historical_normalizing_array['std'])

        assert y_np.shape[0] == x_np.shape[0], f'x and y data have a different ' \
            f'number of instances! x: {x_np.shape[0]}, y: {y_np.shape[0]}'

        if clear_nans:
            # remove nans if they are in the x or y data
            historical_nans, y_nans = np.isnan(train_data.historical), np.isnan(y_np)

            historical_nans_summed = historical_nans.reshape(
                historical_nans.shape[0], historical_nans.shape[1] * historical_nans.shape[2]
            ).sum(axis=-1)
            y_nans_summed = y_nans.sum(axis=-1)

            notnan_indices = np.where((historical_nans_summed == 0) & (y_nans_summed == 0))[0]

            if self.experiment == 'nowcast':
                current_nans = np.isnan(train_data.current)
                current_nans_summed = current_nans.sum(axis=-1)
                notnan_indices = np.where(
                    (
                        historical_nans_summed == 0
                    ) & (y_nans_summed == 0) & (
                        current_nans_summed == 0
                    )
                )[0]
                train_data.current = train_data.current[notnan_indices]  # type: ignore

            train_data.historical, y_np = (
                train_data.historical[notnan_indices], y_np[notnan_indices]
            )
            train_data.pred_months = train_data.pred_months[notnan_indices]  # type: ignore

        if to_tensor:
            train_data.historical, y_np = (
                torch.from_numpy(train_data.historical).float(),
                torch.from_numpy(y_np).float()
            )
            train_data.pred_months = torch.from_numpy(train_data.pred_months).float()

            if self.experiment == 'nowcast':
                train_data.current = torch.from_numpy(train_data.current).float()

        if return_latlons:
            lons, lats = np.meshgrid(x.lon.values, x.lat.values)
            flat_lats, flat_lons = lats.reshape(-1, 1), lons.reshape(-1, 1)
            latlons = np.concatenate((flat_lats, flat_lons), axis=-1)

            if clear_nans:
                latlons = latlons[notnan_indices]

            return ModelArrays(x=train_data, y=y_np, x_vars=list(x.data_vars),
                               y_var=list(y.data_vars)[0], latlons=latlons)

        return ModelArrays(x=train_data, y=y_np, x_vars=list(x.data_vars),
                           y_var=list(y.data_vars)[0])


class _TrainIter(_BaseIter):

    def __next__(self) -> Tuple[Tuple[Union[np.ndarray, torch.Tensor],
                                      Union[np.ndarray, torch.Tensor]],
                                Union[np.ndarray, torch.Tensor]]:

        if self.idx < self.max_idx:
            out_x, out_x_add, out_y = [], [], []

            cur_max_idx = min(self.idx + self.batch_file_size, self.max_idx)
            while self.idx < cur_max_idx:
                subfolder = self.data_files[self.idx]
                arrays = self.ds_folder_to_np(
                    subfolder, clear_nans=self.clear_nans,
                    return_latlons=False, to_tensor=False
                )
                if arrays.x.historical.shape[0] == 0:
                    print(f'{subfolder} returns no values. Skipping')

                    # remove the empty element from the list
                    self.data_files.pop(self.idx)
                    self.max_idx -= 1

                    cur_max_idx = min(cur_max_idx + 1, self.max_idx)

                out_x.append(arrays.x.historical)

                # @GABI we don't want this do we ? it adds a column of constants
                if self.experiment == 'nowcast':
                    # add a constant vector to the X train_data and append
                    constant_vector = np.ones(arrays.x.current.shape)[:, :, :1]  # type: ignore
                    arrays.x.current = np.concatenate(
                        [arrays.x.current, constant_vector], axis=-1
                    )

                out_x_add.append(arrays.x.current)
                out_x_add.append(arrays.x.pred_months)
                out_x_add = [x_add for x_add in out_x_add if x_add is not None]
                out_y.append(arrays.y)
                self.idx += 1

            final_x_add = np.concatenate(out_x_add, axis=0)
            final_x = np.concatenate(out_x, axis=0)
            final_x_add = np.concatenate(out_x_add, axis=0)
            final_y = np.concatenate(out_y, axis=0)

            if final_x.shape[0] == 0:
                raise StopIteration()
            if self.to_tensor:
                return (torch.from_numpy(final_x).float(),
                        torch.from_numpy(final_x_add).float()), torch.from_numpy(final_y).float()
            return (final_x, final_x_add), final_y
        else:
            raise StopIteration()


class _TestIter(_BaseIter):

    def __next__(self) -> Dict[str, ModelArrays]:

        if self.idx < self.max_idx:
            out_dict = {}

            cur_max_idx = min(self.idx + self.batch_file_size, self.max_idx)
            while self.idx < cur_max_idx:
                subfolder = self.data_files[self.idx]
                arrays = self.ds_folder_to_np(subfolder, clear_nans=self.clear_nans,
                                              return_latlons=True, to_tensor=self.to_tensor)
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
