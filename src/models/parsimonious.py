import numpy as np

from typing import Dict, Tuple

from .base import ModelBase
from .data import DataLoader


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

    def predict(
        self,
    ) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, np.ndarray]]:

        test_arrays_loader = DataLoader(
            data_path=self.data_path, batch_file_size=self.batch_size,
            experiment=self.experiment, shuffle_data=False, mode='test', normalize=False,
            pred_months=self.pred_months
        )

        preds_dict: Dict[str, np.ndarray] = {}
        test_arrays_dict: Dict[str, Dict[str, np.ndarray]] = {}
        for dict in test_arrays_loader:
            for key, val in dict.items():
                try:
                    target_idx = val.x_vars.index(val.y_var)
                except ValueError as e:
                    print('Target variable not in prediction data!')
                    raise e

                preds_dict[key] = val.x.historical[:, -1, [target_idx]]
                test_arrays_dict[key] = {'y': val.y, 'latlons': val.latlons}

        return test_arrays_dict, preds_dict
