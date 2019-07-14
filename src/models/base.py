from pathlib import Path
import numpy as np
import json
import pandas as pd
from sklearn.metrics import mean_squared_error

from typing import cast, Any, Dict, List, Optional, Tuple


class ModelBase:
    """Base for all machine learning models.
    Attributes:
    ----------
    data: pathlib.Path = Path('data')
        The location of the data folder.
    batch_size: int 1
        The number of files to load at once. These will be chunked and shuffled, so
        a higher value will lead to better shuffling (but will require more memory)
    pred_months: Optional[List[int]] = None
        The months the model should predict. If None, all months are predicted
    include_pred_month: bool = True
        Whether to include the prediction month to the model's training data
    """

    model_name: str  # to be added by the model classes

    def __init__(self, data_folder: Path = Path('data'),
                 batch_size: int = 1,
                 pred_months: Optional[List[int]] = None,
                 include_pred_month: bool = True,
                 surrounding_pixels: Optional[int] = None) -> None:

        self.batch_size = batch_size
        self.include_pred_month = include_pred_month
        self.data_path = data_folder
        self.pred_months = pred_months
        self.surrounding_pixels = surrounding_pixels

        self.models_dir = data_folder / 'models'
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

    def predict(self) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, np.ndarray]]:
        # This method should return the test arrays as loaded by
        # the test array dataloader, and the corresponding predictions
        raise NotImplementedError

    def explain(self, x: Any) -> np.ndarray:
        """
        Explain the predictions of the trained model on the input data x

        Arguments
        ----------
        x: Any
            An input array / tensor

        Returns
        ----------
        explanations: np.ndarray
            A shap value for each of the input values. The sum of the shap
            values is equal to the prediction of the model for x
        """
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
        for key, vals in test_arrays_dict.items():
            true = vals['y']
            preds = preds_dict[key]

            output_dict[key] = np.sqrt(mean_squared_error(true, preds)).item()

            total_preds.append(preds)
            total_true.append(true)

        output_dict['total'] = np.sqrt(mean_squared_error(np.concatenate(total_true),
                                                          np.concatenate(total_preds))).item()

        print(f'RMSE: {output_dict["total"]}')

        if save_results:
            with (self.model_dir / 'results.json').open('w') as outfile:
                json.dump(output_dict, outfile)

        if save_preds:
            for key, val in test_arrays_dict.items():
                latlons = cast(np.ndarray, val['latlons'])
                preds = preds_dict[key]

                if len(preds.shape) > 1:
                    preds = preds.squeeze(-1)

                preds_xr = pd.DataFrame(data={
                    'preds': preds, 'lat': latlons[:, 0],
                    'lon': latlons[:, 1]}).set_index(['lat', 'lon']).to_xarray()

                preds_xr.to_netcdf(self.model_dir / f'preds_{key}.nc')
