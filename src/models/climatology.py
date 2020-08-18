import numpy as np
import xarray as xr

from typing import Dict, Tuple

from .base import ModelBase
from src.analysis import read_train_data, read_test_data


class Climatology(ModelBase):
    """A parsimonious Climatology model.
    This "model" predicts the mean value for that month. For example, its prediction
    for VHI in March 2018 will be mean VHI in March across all training data
    """

    model_name = "climatology"

    def train(self) -> None:
        pass

    def save_model(self) -> None:
        print("Move on! Nothing to save here!")

    def predict(
        self, all_data: bool = False
    ) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, np.ndarray]]:

        _, y_train = read_train_data(self.data_path)
        ds = y_train

        if all_data:
            # if want to calculate climatology for train+test data
            _, y_test = read_test_data(self.data_path)
            ds = xr.merge([y_train, y_test]).sortby("time").sortby("lat")

        target_var = [v for v in ds.data_vars][0]

        # calculate climatology:
        monmean = ds.groupby("time.month").mean(dim=["time"])[target_var]
        monmean = monmean.stack(pixel=["lat", "lon"])

        test_arrays_loader = self.get_dataloader(
            mode="test", shuffle_data=False, normalize=False, static=False
        )

        preds_dict: Dict[str, np.ndarray] = {}
        test_arrays_dict: Dict[str, Dict[str, np.ndarray]] = {}

        for dict in test_arrays_loader:
            for key, val in dict.items():
                try:
                    _ = val.x_vars.index(val.y_var)
                except ValueError as e:
                    print("Target variable not in prediction data!")
                    raise e

                # TODO: remove the missing data from the array
                # the same way that the test_arrays_loader does
                climatology_vals = monmean.sel(
                    month=val.target_time.month
                ).values.flatten()

                # get the not nan indices
                climatology_vals = climatology_vals[val.notnan_indices]

                preds_dict[key] = climatology_vals.reshape(val.y.shape)

                test_arrays_dict[key] = {
                    "y": val.y,
                    "latlons": val.latlons,
                    "time": val.target_time,
                    "y_var": val.y_var,
                    "nan_mask": val.nan_mask,
                }

        return test_arrays_dict, preds_dict
