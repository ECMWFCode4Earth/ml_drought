import numpy as np

from typing import Dict, Tuple

from .base import ModelBase, ModelArrays


class Persistence(ModelBase):
    """A parsimonious persistence model.

    This "model" predicts the previous time-value of data. For example, its prediction
    for VHI in March 2018 will be VHI for February 2018 (assuming monthly time-granularity).
    """

    model_name = 'previous_month'

    def train(self) -> None:
        pass

    def save_model(self) -> None:
        print('Move on! Nothing to save here!')

    def predict(self) -> Tuple[Dict[str, ModelArrays], Dict[str, np.ndarray]]:

        test_arrays = self.load_test_arrays()

        preds_dict: Dict[str, np.ndarray] = {}
        for key, val in test_arrays.items():

            try:
                target_idx = val.x_vars.index(val.y_var)
            except ValueError as e:
                print('Target variable not in prediction data!')
                raise e

            preds_dict[key] = val.x[:, -1, [target_idx]]

        return test_arrays, preds_dict
