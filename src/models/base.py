from pathlib import Path
import numpy as np
import json
import pandas as pd
import itertools
from sklearn.metrics import mean_squared_error

from typing import cast, Any, Dict, List, Optional, Tuple, Union


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
    surrounding_pixels: Optional[int] = None
        How many surrounding pixels to add to the input data. e.g. if the input is 1, then in
        addition to the pixels on the prediction point, the neighbouring (spatial) pixels will
        be included too, up to a distance of one pixel away
    ignore_vars: Optional[List[str]] = None
        A list of variables to ignore. If None, all variables in the data_path will be included
    include_latlons: bool = True
        Whether to include prediction pixel latitudes and longitudes in the model's
        training data
    include_static: bool = True
        Whether to include static data
    """

    model_name: str  # to be added by the model classes

    def __init__(self, data_folder: Path = Path('data'),
                 batch_size: int = 1,
                 experiment: str = 'one_month_forecast',
                 pred_months: Optional[List[int]] = None,
                 include_pred_month: bool = True,
                 include_latlons: bool = False,
                 include_monthly_aggs: bool = True,
                 include_yearly_aggs: bool = True,
                 surrounding_pixels: Optional[int] = None,
                 ignore_vars: Optional[List[str]] = None,
                 include_static: bool = True) -> None:

        self.batch_size = batch_size
        self.include_pred_month = include_pred_month
        self.include_latlons = include_latlons
        self.include_monthly_aggs = include_monthly_aggs
        self.include_yearly_aggs = include_yearly_aggs
        self.data_path = data_folder
        self.experiment = experiment
        self.pred_months = pred_months
        self.models_dir = data_folder / 'models' / self.experiment
        self.surrounding_pixels = surrounding_pixels
        self.ignore_vars = ignore_vars
        self.include_static = include_static

        if not self.models_dir.exists():
            self.models_dir.mkdir(parents=True, exist_ok=False)

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

    def _run_evaluation_calculation(
        self,
        test_arrays_dict: Dict[str, Dict[str, np.ndarray]],
        preds_dict: Dict[str, np.ndarray]
    ) -> Union[Dict[str, int], List[np.ndarray], List[np.ndarray]]:
        """Calculate RMSE for the true vs. predicted values"""
        output_dict: Dict[str, int] = {}
        total_preds: List[np.ndarray] = []
        total_true: List[np.ndarray] = []
        for key, vals in test_arrays_dict.items():
            true = vals['y']
            preds = preds_dict[key]

            output_dict[key] = np.sqrt(mean_squared_error(true, preds)).item()

            total_preds.append(preds)
            total_true.append(true)

        return output_dict, total_preds, total_true

    def _save_preds(self,
                    test_arrays_dict: Dict[str, Dict[str, np.ndarray]],
                    preds_dict: Dict[str, np.ndarray],
                    train: bool = False) -> None:
        for key, val in test_arrays_dict.items():
            latlons = cast(np.ndarray, val['latlons'])
            preds = preds_dict[key]

            if len(preds.shape) > 1:
                preds = preds.squeeze(-1)

            # the prediction timestep
            time = val['time']
            times = [time for _ in range(len(preds))]

            preds_xr = pd.DataFrame(data={
                'preds': preds, 'lat': latlons[:, 0],
                'lon': latlons[:, 1], 'time': times}
            ).set_index(['lat', 'lon', 'time']).to_xarray()

            if train:
                if not (self.model_dir / 'train').exists():
                    (self.model_dir / 'train').mkdir(exist_ok=True, parents=True)
                preds_xr.to_netcdf(self.model_dir / 'train' / f'preds_{key}.nc')
            else:
                preds_xr.to_netcdf(self.model_dir / f'preds_{key}.nc')

    def evaluate_train_timesteps(self, years: List[int],
                                 months: List[int],
                                 save_results: bool = True,
                                 save_preds: bool = False) -> None:
        """Evaluate the trained model on the TRAIN data
        for the given year / month combinations.
        """
        all_total_true: List[np.ndarray] = []
        all_total_preds: List[np.ndarray] = []
        all_rmse_dict: Dict[str, int] = {}
        for test_year, test_month in itertools.product(years, months):
            test_arrays_dict, preds_dict = self.predict(
                test_year=test_year, test_month=test_month
            )
            output_dict, total_preds, total_true = self._run_evaluation_calculation(
                test_arrays_dict=test_arrays_dict,
                preds_dict=preds_dict,
            )
            all_total_true.append(total_preds)
            all_total_preds.append(total_true)
            if save_results:
                if not (self.model_dir / 'train').exists():
                    (self.model_dir / 'train').mkdir(exist_ok=True, parents=True)
                with (
                    self.model_dir / 'train' /
                    f'results_{test_year}_{test_month}.json'
                ).open('w') as outfile:
                    json.dump(output_dict, outfile)

            if save_preds:
                self._save_preds(
                    test_arrays_dict=test_arrays_dict,
                    preds_dict=preds_dict
                )

        # calculate the RMSE for all the training timesteps
        all_rmse_dict['total'] = np.sqrt(
            mean_squared_error(np.concatenate(all_total_true).flatten(),
                               np.concatenate(all_total_preds).flatten())
        ).item()
        print(
            f'RMSE for given timesteps ({years} - {months}): '
            f'{all_rmse_dict["total"]}'
        )

        if save_results:
            with (self.model_dir / 'train' / 'results.json').open('w') as outfile:
                json.dump(all_rmse_dict, outfile)

    def evaluate(self, save_results: bool = True,
                 save_preds: bool = False) -> None:
        """
        Evaluate the trained model on the TEST data

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
        output_dict, total_preds, total_true = self._run_evaluation_calculation(
            test_arrays_dict=test_arrays_dict,
            preds_dict=preds_dict,
        )

        output_dict['total'] = np.sqrt(
            mean_squared_error(np.concatenate(total_true),
                               np.concatenate(total_preds))
        ).item()

        print(f'RMSE: {output_dict["total"]}')

        if save_results:
            with (self.model_dir / 'results.json').open('w') as outfile:
                json.dump(output_dict, outfile)

        if save_preds:
            self._save_preds(
                test_arrays_dict=test_arrays_dict,
                preds_dict=preds_dict
            )
