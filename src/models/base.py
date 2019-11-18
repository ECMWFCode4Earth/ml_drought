from pathlib import Path
import numpy as np
import json
import pandas as pd
from sklearn.metrics import mean_squared_error

from .data import TrainData, DataLoader

from typing import cast, Any, Dict, List, Optional, Union, Tuple


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

    def __init__(
        self,
        data_folder: Path = Path("data"),
        batch_size: int = 1,
        experiment: str = "one_month_forecast",
        pred_months: Optional[List[int]] = None,
        include_pred_month: bool = True,
        include_latlons: bool = False,
        include_monthly_aggs: bool = True,
        include_yearly_aggs: bool = True,
        surrounding_pixels: Optional[int] = None,
        ignore_vars: Optional[List[str]] = None,
        include_static: bool = True,
    ) -> None:

        self.batch_size = batch_size
        self.include_pred_month = include_pred_month
        self.include_latlons = include_latlons
        self.include_monthly_aggs = include_monthly_aggs
        self.include_yearly_aggs = include_yearly_aggs
        self.data_path = data_folder
        self.experiment = experiment
        self.pred_months = pred_months
        self.models_dir = data_folder / "models" / self.experiment
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
            print(
                "Model name attribute must be defined for the model directory to be created"
            )

        self.model: Any = None  # to be added by the model classes
        self.data_vars: Optional[List[str]] = None  # to be added by the train step

        # This can be overridden by any model which actually cares which device its run on
        # by default, models which don't care will run on the CPU
        self.device = "cpu"

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

        output_dict: Dict[str, int] = {}
        total_preds: List[np.ndarray] = []
        total_true: List[np.ndarray] = []
        for key, vals in test_arrays_dict.items():
            true = vals["y"]
            preds = preds_dict[key]

            output_dict[key] = np.sqrt(mean_squared_error(true, preds)).item()

            total_preds.append(preds)
            total_true.append(true)

        output_dict["total"] = np.sqrt(
            mean_squared_error(np.concatenate(total_true), np.concatenate(total_preds))
        ).item()

        print(f'RMSE: {output_dict["total"]}')

        if save_results:
            with (self.model_dir / "results.json").open("w") as outfile:
                json.dump(output_dict, outfile)

        if save_preds:
            for key, val in test_arrays_dict.items():
                latlons = cast(np.ndarray, val["latlons"])
                preds = preds_dict[key]

                if len(preds.shape) > 1:
                    preds = preds.squeeze(-1)

                # the prediction timestep
                time = val["time"]
                times = [time for _ in range(len(preds))]

                preds_xr = (
                    pd.DataFrame(
                        data={
                            "preds": preds,
                            "lat": latlons[:, 0],
                            "lon": latlons[:, 1],
                            "time": times,
                        }
                    )
                    .set_index(["lat", "lon", "time"])
                    .to_xarray()
                )

                preds_xr.to_netcdf(self.model_dir / f"preds_{key}.nc")

    def _concatenate_data(
        self, x: Union[Tuple[Optional[np.ndarray], ...], TrainData]
    ) -> np.ndarray:
        """Takes a TrainData object, and flattens all the features the model
        is using as predictors into a np.ndarray
        """

        if type(x) is tuple:
            x_his, x_pm, x_latlons, x_cur, x_ym, x_static = x  # type: ignore
        elif type(x) == TrainData:
            x_his, x_pm, x_latlons = x.historical, x.pred_months, x.latlons  # type: ignore
            x_cur, x_ym = x.current, x.yearly_aggs  # type: ignore
            x_static = x.static  # type: ignore

        assert (
            x_his is not None
        ), "x[0] should be historical data, and therefore should not be None"
        x_in = x_his.reshape(x_his.shape[0], x_his.shape[1] * x_his.shape[2])

        if self.include_pred_month:
            # one hot encoding, should be num_classes + 1, but
            # for us its + 2, since 0 is not a class either
            pred_months_onehot = np.eye(14)[x_pm][:, 1:-1]
            x_in = np.concatenate((x_in, pred_months_onehot), axis=-1)
        if self.include_latlons:
            x_in = np.concatenate((x_in, x_latlons), axis=-1)
        if self.experiment == "nowcast":
            x_in = np.concatenate((x_in, x_cur), axis=-1)
        if self.include_yearly_aggs:
            x_in = np.concatenate((x_in, x_ym), axis=-1)
        if self.include_static:
            x_in = np.concatenate((x_in, x_static), axis=-1)

        return x_in

    def get_dataloader(
        self, mode: str, to_tensor: bool = False, shuffle_data: bool = False, **kwargs
    ) -> DataLoader:
        """
        Return the correct dataloader for this model
        """

        default_args: Dict[str, Any] = {
            "data_path": self.data_path,
            "batch_file_size": self.batch_size,
            "shuffle_data": shuffle_data,
            "mode": mode,
            "mask": None,
            "experiment": self.experiment,
            "pred_months": self.pred_months,
            "to_tensor": to_tensor,
            "ignore_vars": self.ignore_vars,
            "monthly_aggs": self.include_monthly_aggs,
            "surrounding_pixels": self.surrounding_pixels,
            "static": self.include_static,
            "device": self.device,
            "clear_nans": True,
            "normalize": True,
        }

        for key, val in kwargs.items():
            # override the default args
            default_args[key] = val

        return DataLoader(**default_args)
