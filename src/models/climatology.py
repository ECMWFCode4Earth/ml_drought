import numpy as np
import xarray as xr

from typing import Dict, Tuple

from .base import ModelBase
from src.analysis import read_train_data, read_test_data


class Persistence(ModelBase):
    """A parsimonious persistence model.
    This "model" predicts the mean value for that month. For example, its prediction
    for VHI in March 2018 will be mean VHI in March across all training data
    """

    model_name = "climatology"


    def train(self) -> None:
        pass

    def save_model(self) -> None:
        print("Move on! Nothing to save here!")

    def predict(
        self,
        all_data: bool = False
    ) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, np.ndarray]]:

        _, y_train = read_train_data(data_dir)
        ds = y_train

        if all_data:
            # if want to calculate climatology for train+test data
            _, y_test = read_test_data(data_dir)
            ds = xr.merge([y_train, y_test]).sortby('time').sortby('lat')

        target_var = [v for v in ds.data_vars][0]
        monmean = ds.groupby('time.month').mean().target_var

        test_arrays_loader = self.get_dataloader(
            mode="test", shuffle_data=False, normalize=False, static=False
        )

        preds_dict: Dict[str, np.ndarray] = {}
        test_arrays_dict: Dict[str, Dict[str, np.ndarray]] = {}
        for dict in test_arrays_loader:
            for key, val in dict.items():
                try:
                    target_idx = val.x_vars.index(val.y_var)
                except ValueError as e:
                    print("Target variable not in prediction data!")
                    raise e

                preds_dict[key] = monmean.sel(time=val.target_time).values
                test_arrays_dict[key] = {
                    "y": val.y,
                    "latlons": val.latlons,
                    "time": val.target_time,
                    "y_var": val.y_var,
                }

        return test_arrays_dict, preds_dict
