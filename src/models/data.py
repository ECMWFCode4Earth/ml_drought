from dataclasses import dataclass
import numpy as np
from random import shuffle
from pathlib import Path
import xarray as xr

from typing import Dict, Optional, List, Tuple


@dataclass
class ModelArrays:
    x: np.ndarray
    y: np.ndarray
    x_vars: List[str]
    y_var: str
    latlons: Optional[np.ndarray] = None


class DataLoader:
    """Base class for the train and test dataloaders
    """
    def __init__(self, data_path: Path = Path('data'), batch_file_size: int = 1,
                 mode: str = 'train', shuffle_data: bool = True,
                 clear_nans: bool = True) -> None:

        self.batch_file_size = batch_file_size
        self.mode = mode
        self.shuffle = shuffle_data
        self.clear_nans = clear_nans
        self.data_files = self._load_datasets(data_path, mode, shuffle_data)

    def __iter__(self):
        if self.mode == 'train':
            return _TrainIter(self)
        else:
            return _TestIter(self)

    def __len__(self) -> int:
        return len(self.data_files) // self.batch_file_size

    @staticmethod
    def _load_datasets(data_path: Path, mode: str, shuffle_data: bool) -> List[Path]:

        data_folder = data_path / f'features/{mode}'
        output_paths: List[Path] = []

        for subtrain in data_folder.iterdir():
            if (subtrain / 'x.nc').exists() and (subtrain / 'y.nc').exists():
                output_paths.append(subtrain)
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

        self.idx = 0
        self.max_idx = len(loader.data_files)

    def __iter__(self):
        return self

    @staticmethod
    def ds_folder_to_np(folder: Path,
                        clear_nans: bool = True,
                        return_latlons: bool = False,
                        ) -> ModelArrays:

        x, y = xr.open_dataset(folder / 'x.nc'), xr.open_dataset(folder / 'y.nc')
        assert len(list(y.data_vars)) == 1, f'Expect only 1 target variable!'
        x_np, y_np = x.to_array().values, y.to_array().values

        # first, x
        x_np = x_np.reshape(x_np.shape[0], x_np.shape[1], x_np.shape[2] * x_np.shape[3])
        x_np = np.moveaxis(np.moveaxis(x_np, 0, 1), -1, 0)
        # then, y
        y_np = y_np.reshape(y_np.shape[0], y_np.shape[1], y_np.shape[2] * y_np.shape[3])
        y_np = np.moveaxis(y_np, -1, 0).reshape(-1, 1)

        assert y_np.shape[0] == x_np.shape[0], f'x and y data have a different ' \
            f'number of instances! x: {x_np.shape[0]}, y: {y_np.shape[0]}'

        if clear_nans:
            # remove nans if they are in the x or y data
            x_nans, y_nans = np.isnan(x_np), np.isnan(y_np)

            x_nans_summed = x_nans.reshape(x_nans.shape[0],
                                           x_nans.shape[1] * x_nans.shape[2]).sum(axis=-1)
            y_nans_summed = y_nans.sum(axis=-1)

            notnan_indices = np.where((x_nans_summed == 0) & (y_nans_summed == 0))[0]
            x_np, y_np = x_np[notnan_indices], y_np[notnan_indices]

        if return_latlons:
            lons, lats = np.meshgrid(x.lon.values, x.lat.values)
            flat_lats, flat_lons = lats.reshape(-1, 1), lons.reshape(-1, 1)
            latlons = np.concatenate((flat_lats, flat_lons), axis=-1)

            if clear_nans:
                latlons = latlons[notnan_indices]
            return ModelArrays(x=x_np, y=y_np, x_vars=list(x.data_vars),
                               y_var=list(y.data_vars)[0], latlons=latlons)

        return ModelArrays(x=x_np, y=y_np, x_vars=list(x.data_vars),
                           y_var=list(y.data_vars)[0])


class _TrainIter(_BaseIter):

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:

        if self.idx < self.max_idx:
            out_x, out_y = [], []

            cur_max_idx = min(self.idx + self.batch_file_size, self.max_idx)
            for subfolder in self.data_files[self.idx: cur_max_idx]:
                arrays = self.ds_folder_to_np(subfolder, clear_nans=self.clear_nans,
                                              return_latlons=False)
                out_x.append(arrays.x)
                out_y.append(arrays.y)

            self.idx = self.idx + self.batch_file_size
            return np.concatenate(out_x, axis=0), np.concatenate(out_y, axis=0)
        else:
            raise StopIteration()


class _TestIter(_BaseIter):

    def __next__(self) -> Dict[str, ModelArrays]:

        if self.idx < self.max_idx:
            out_dict = {}

            cur_max_idx = min(self.idx + self.batch_file_size, self.max_idx)
            for subtrain in self.data_files[self.idx: cur_max_idx]:
                modelarrays = self.ds_folder_to_np(subtrain, clear_nans=self.clear_nans,
                                                   return_latlons=True)
                out_dict[subtrain.parts[-1]] = modelarrays

            self.idx = self.idx + self.batch_file_size
            return out_dict
        else:
            raise StopIteration()
