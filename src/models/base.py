from pathlib import Path
import numpy as np
import json
import xarray as xr
import pandas as pd
from dataclasses import dataclass
from sklearn.metrics import mean_squared_error

from typing import cast, Any, Dict, List, Optional, Tuple


@dataclass
class ModelArrays:
    x: np.ndarray
    y: np.ndarray
    x_vars: List[str]
    y_var: str
    latlons: Optional[np.ndarray] = None


class ModelBase:
    """Base for all machine learning models.
    Attributes:
    ----------
    data: pathlib.Path
        The location of the data folder.
    """

    model_name: str  # to be added by the model classes

    def __init__(self, data: Path = Path('data')):

        self.data_path = data
        self.models_dir = data / 'models'
        if not self.models_dir.exists():
            self.models_dir.mkdir()

        try:
            self.model_dir = self.models_dir / self.model_name
            if not self.model_dir.exists():
                self.model_dir.mkdir()
        except AttributeError:
            print('Model name attribute must be defined for the model directory to be created')

        self.model: Any = None  # to be added by the model classes
        self.data_vars: Optional[List[str]] = None  # to be added by the train step

    def train(self) -> None:
        raise NotImplementedError

    def predict(self) -> Tuple[Dict[str, ModelArrays], Dict[str, np.ndarray]]:
        # This method should return the test arrays as loaded by
        # load_test_arrays, and the corresponding predictions
        raise NotImplementedError

    def save_model(self) -> None:
        raise NotImplementedError

    def evaluate(self, save_results: bool = True, save_preds: bool = False) -> None:
        """
        Evaluate the trained model

        Arguments
        ----------
        save_results: bool = True
            Whether to save the results of the evaluation. If true, they are
            saved in self.model_dir / results.json
        save_preds: bool = False
            Whether to save the model predictions. If true, they are saved in
            self.model_dir / {year}_{month}.nc
        """
        test_arrays_dict, preds_dict = self.predict()

        output_dict: Dict[str, int] = {}
        total_preds: List[np.ndarray] = []
        total_true: List[np.ndarray] = []
        for key, val in test_arrays_dict.items():
            preds = preds_dict[key]
            true = val.y

            output_dict[key] = np.sqrt(mean_squared_error(true, preds))

            total_preds.append(preds)
            total_true.append(true)

        output_dict['total'] = np.sqrt(mean_squared_error(np.concatenate(total_true),
                                                          np.concatenate(total_preds)))

        print(f'RMSE: {output_dict["total"]}')

        if save_results:
            with (self.model_dir / 'results.json').open('w') as outfile:
                json.dump(output_dict, outfile)

        if save_preds:
            for key, val in test_arrays_dict.items():
                latlons = cast(np.ndarray, val.latlons)
                preds = preds_dict[key]

                if len(preds.shape) > 1:
                    preds = preds.squeeze(-1)

                preds_xr = pd.DataFrame(data={
                    'preds': preds, 'lat': latlons[:, 0],
                    'lon': latlons[:, 1]}).set_index(['lat', 'lon']).to_xarray()

                preds_xr.to_netcdf(self.model_dir / f'preds_{key}.nc')

    def load_test_arrays(self) -> Dict[str, ModelArrays]:
        test_data_path = self.data_path / 'features/test'

        out_dict = {}
        for subtrain in test_data_path.iterdir():
            if (subtrain / 'x.nc').exists() and (subtrain / 'y.nc').exists():
                out_dict[subtrain.parts[-1]] = self.ds_folder_to_np(subtrain, clear_nans=True,
                                                                    return_latlons=True)
        return out_dict

    def load_train_arrays(self) -> Tuple[np.ndarray, np.ndarray]:

        train_data_path = self.data_path / 'features/train'

        out_x, out_y = [], []
        for subtrain in train_data_path.iterdir():
            if (subtrain / 'x.nc').exists() and (subtrain / 'y.nc').exists():
                arrays = self.ds_folder_to_np(subtrain, clear_nans=True,
                                              return_latlons=False)
                out_x.append(arrays.x)
                out_y.append(arrays.y)
        return np.concatenate(out_x, axis=0), np.concatenate(out_y, axis=0)

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
