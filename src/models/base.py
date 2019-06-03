from pathlib import Path
import numpy as np
import xarray as xr

from typing import cast, List, Optional, Tuple


class ModelBase:
    """Base for all machine learning models.
    Attributes:
    ----------
    data: pathlib.Path
        The location of the data folder.
    """

    model_name: Optional[str] = None  # to be added by the model classes

    def __init__(self, data: Path = Path('data')):

        self.data_path = data
        self.models_dir = data / 'models'
        if not self.models_dir.exists():
            self.models_dir.mkdir()

        try:
            self.model_dir = self.models_dir / cast(str, self.model_name)
            if not self.model_dir.exists():
                self.model_dir.mkdir()
        except AttributeError:
            print('Model name attribute must be defined for the model directory to be created')

        self.model: Optional[str] = None  # to be added by the model classes
        self.data_vars: Optional[List[str]] = None  # to be added by the train step

    def train(self):
        raise NotImplementedError

    def predict(self):
        # This method should return the predictions, and
        # the corresponding true values, read from the test
        # arrays
        raise NotImplementedError

    def save_model(self):
        # This method should save the model in data / model_name
        raise NotImplementedError

    def evaluate(self, return_eval=False, save_preds=False):
        # TODO
        raise NotImplementedError

    def load_test_arrays(self):
        # TODO
        raise NotImplementedError

    @staticmethod
    def ds_folder_to_np(folder: Path,
                        clear_nans: bool = True,
                        return_latlons: bool = False,
                        ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:

        x, y = xr.open_dataset(folder / 'x.nc'), xr.open_dataset(folder / 'y.nc')
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

        latlons = None
        if return_latlons:
            lons, lats = np.meshgrid(x.lon.values, x.lat.values)
            flat_lats, flat_lons = lats.reshape(-1, 1), lons.reshape(-1, 1)
            latlons = np.concatenate((flat_lats, flat_lons), axis=-1)

            if clear_nans:
                latlons = latlons[notnan_indices]

        return x_np, y_np, latlons
