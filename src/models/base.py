from pathlib import Path
import pickle
import numpy as np
import json
import pandas as pd
import xarray as xr
import random
from sklearn.metrics import mean_squared_error, r2_score

from .data import TrainData, DataLoader
from .utils import vals_dict_to_xarray_dataset

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
    predict_delta: bool = False
        Whether to model the CHANGE in target variable rather than the
        raw values
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
        static: Optional[str] = "embeddings",
        predict_delta: bool = False,
        spatial_mask: Union[xr.DataArray, Path] = None,
        include_prev_y: bool = True,
        normalize_y: bool = False,
        clear_nans: bool = True,
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
        self.static = static
        self.predict_delta = predict_delta
        self.include_prev_y = include_prev_y
        self.normalize_y = normalize_y
        self.clear_nans = clear_nans
        if normalize_y:
            with (data_folder / f"features/{experiment}/normalizing_dict.pkl").open(
                "rb"
            ) as f:
                self.normalizing_dict = pickle.load(f)

        # needs to be set by the train function
        self.num_locations: Optional[int] = None

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
        self.spatial_mask = self._load_spatial_mask(spatial_mask)

        # This can be overridden by any model which actually cares which device its run on
        # by default, models which don't care will run on the CPU
        self.device = "cpu"
        np.random.seed(42)
        random.seed(42)

    @staticmethod
    def _load_spatial_mask(
        mask: Union[Path, xr.DataArray, None] = None
    ) -> Optional[xr.DataArray]:
        if (mask is None) or isinstance(mask, xr.DataArray):
            return mask
        elif isinstance(mask, Path):
            mask = xr.open_dataset(mask)
            return mask["mask"]
        return None

    def _convert_delta_to_raw_values(
        self, x: xr.Dataset, y: xr.Dataset, y_var: str, order: int = 1
    ) -> xr.Dataset:
        """When calculating the derivative we need to convert the change/delta
        to the raw value for our prediction.
        """
        # x.shape == (pixels, featurespreds)
        prev_ts = x[y_var].isel(time=-order)

        # calculate the raw values
        return prev_ts + y["preds"]  # .to_dataset(name_of_preds_var)

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

    def denormalize_y(self, y: np.ndarray, var_name: str) -> np.ndarray:

        if not self.normalize_y:
            return y
        else:
            y = y * self.normalizing_dict[var_name]["std"]

            if not self.predict_delta:
                y = y + self.normalizing_dict[var_name]["mean"]
        return y

    def evaluate(
        self,
        save_results: bool = True,
        save_preds: bool = False,
        check_inverted: bool = False,
    ) -> None:
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
        check_inverted: bool = False
            Whether to check if the models are somewhere inverting the data
            (boolean switch because it can slow the code down)
        """
        test_arrays_dict, preds_dict = self.predict()

        output_dict: Dict[str, int] = {}
        total_preds: List[np.ndarray] = []
        total_true: List[np.ndarray] = []
        for key, vals in test_arrays_dict.items():

            true = self.denormalize_y(vals["y"], vals["y_var"])
            preds = self.denormalize_y(preds_dict[key], vals["y_var"])

            if check_inverted:
                # turn into xarray objects
                pred_ds = vals_dict_to_xarray_dataset(vals, preds, "preds")
                true_ds = vals_dict_to_xarray_dataset(vals, true, "y_true")
                # check that the shapes are similar / same?
                # TODO: how to do this in a general way ...? Choice of threshold
                # check that the matching missing values are ~0.92
                # (this catches the inversion problem)
                assert (
                    pred_ds.isnull().preds.values == true_ds.isnull().y_true.values
                ).mean() > 0.92, (
                    "The missing data should be the same for 92% of the data. "
                    "This sometimes occurs when there has been a problem with an inversion "
                    "somewhere in the data"
                )

            output_dict[key] = np.sqrt(mean_squared_error(true, preds)).item()

            total_preds.append(preds)
            total_true.append(true)

        output_dict["total"] = np.sqrt(
            mean_squared_error(np.concatenate(total_true), np.concatenate(total_preds))
        ).item()
        r2 = r2_score(np.concatenate(total_true), np.concatenate(total_preds))
        print(f'RMSE: {output_dict["total"]}')
        print(f"R2: {r2:.2f}")

        if save_results:
            with (self.model_dir / "results.json").open("w") as outfile:
                json.dump(output_dict, outfile)

        if save_preds:
            # convert from test_arrays_dict to xarray object
            for key, val in test_arrays_dict.items():
                latlons = cast(np.ndarray, val["latlons"])
                preds = self.denormalize_y(preds_dict[key], val["y_var"])

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

                if self.predict_delta:
                    # get the NON-NORMALIZED data (from ModelArrays.historical_target)
                    historical_y = val["historical_target"]
                    y_var = val["y_var"]
                    cast(str, y_var)
                    historical_ds = (
                        pd.DataFrame(
                            data={
                                y_var: historical_y,
                                "lat": latlons[:, 0],
                                "lon": latlons[:, 1],
                                "time": times,
                            }
                        )
                        .set_index(["lat", "lon", "time"])
                        .to_xarray()
                    )

                    # convert delta to raw target_variable
                    preds_xr = self._convert_delta_to_raw_values(
                        x=historical_ds, y=preds_xr, y_var=y_var
                    )

                    if not isinstance(preds_xr, xr.Dataset):
                        preds_xr = preds_xr.to_dataset("preds")

                preds_xr.to_netcdf(self.model_dir / f"preds_{key}.nc")

    def _concatenate_data(
        self, x: Union[Tuple[Optional[np.ndarray], ...], TrainData]
    ) -> np.ndarray:
        """Takes a TrainData object, and flattens all the features the model
        is using as predictors into a np.ndarray
        """

        if type(x) is tuple:
            x_his, x_pm, x_latlons, x_cur, x_ym, x_static, x_prev = x  # type: ignore
        elif type(x) == TrainData:
            x_his, x_pm, x_latlons = (
                x.historical,  # type: ignore
                x.pred_months,  # type: ignore
                x.latlons,  # type: ignore
            )  # type: ignore
            x_cur, x_ym = x.current, x.yearly_aggs  # type: ignore
            x_static = x.static  # type: ignore
            x_prev = x.prev_y_var  # type: ignore

        assert (
            x_his is not None
        ), "x[0] should be historical data, and therefore should not be None"
        x_in = x_his.reshape(x_his.shape[0], x_his.shape[1] * x_his.shape[2])

        if self.include_pred_month:
            # one hot encoding, should be num_classes + 1, but
            # for us its + 2, since 0 is not a class either
            pred_months_onehot = self._one_hot(x_pm, 12)
            x_in = np.concatenate((x_in, pred_months_onehot), axis=-1)
        if self.include_latlons:
            x_in = np.concatenate((x_in, x_latlons), axis=-1)
        if self.experiment == "nowcast":
            x_in = np.concatenate((x_in, x_cur), axis=-1)
        if self.include_yearly_aggs:
            x_in = np.concatenate((x_in, x_ym), axis=-1)
        if self.static is not None:
            if self.static == "features":
                x_in = np.concatenate((x_in, x_static), axis=-1)
            elif self.static == "embeddings":
                assert type(self.num_locations) is int
                x_s = self._one_hot(x_static, cast(int, self.num_locations))
                x_in = np.concatenate((x_in, x_s), axis=-1)
        if self.include_prev_y:
            x_in = np.concatenate((x_in, x_prev), axis=-1)
        return x_in

    def _one_hot(self, x: np.ndarray, num_vals: int):
        if len(x.shape) > 1:
            x = x.squeeze(-1)
        return np.eye(num_vals + 2)[x][:, 1:-1]

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
            "incl_yearly_aggs": self.include_yearly_aggs,
            "surrounding_pixels": self.surrounding_pixels,
            "static": self.static,
            "device": self.device,
            "clear_nans": self.clear_nans,
            "normalize": True,
            "predict_delta": self.predict_delta,
            "spatial_mask": self.spatial_mask,
            "normalize_y": self.normalize_y,
        }

        for key, val in kwargs.items():
            # override the default args
            default_args[key] = val

        dl = DataLoader(**default_args)

        if (self.static == "embeddings") and (self.num_locations is None):
            self.num_locations = cast(int, dl.max_loc_int)
        return dl
