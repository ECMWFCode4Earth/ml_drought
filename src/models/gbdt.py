import numpy as np
from pathlib import Path
import pickle
import xarray as xr

from typing import Any, Dict, List, Tuple, Optional, Union

from .base import ModelBase
from .data import DataLoader, train_val_mask, TrainData

xgb = None


class GBDT(ModelBase):
    """Trains a GBDT, using the XGBoost implementation.

    The XGBoost regressor requires all the training data to
    be in memory, so this model needs to be trained on a very
    big machine, or on a subset of the data.
    """

    model_name = "gbdt"

    def __init__(
        self,
        data_folder: Path = Path("data"),
        experiment: str = "one_month_forecast",
        batch_size: int = 1,
        pred_months: Optional[List[int]] = None,
        include_pred_month: bool = True,
        include_latlons: bool = False,
        include_monthly_aggs: bool = True,
        include_yearly_aggs: bool = True,
        surrounding_pixels: Optional[int] = None,
        ignore_vars: Optional[List[str]] = None,
        static: Optional[str] = "features",
        predict_delta: bool = False,
        spatial_mask: Union[xr.DataArray, Path] = None,
        include_prev_y: bool = True,
        normalize_y: bool = True,
    ) -> None:
        super().__init__(
            data_folder,
            batch_size,
            experiment,
            pred_months,
            include_pred_month,
            include_latlons,
            include_monthly_aggs,
            include_yearly_aggs,
            surrounding_pixels,
            ignore_vars,
            static,
            predict_delta=predict_delta,
            spatial_mask=spatial_mask,
            include_prev_y=include_prev_y,
            normalize_y=normalize_y,
        )

        self.early_stopping = False

        global xgb
        if xgb is None:
            import xgboost as xgb

    def train(
        self, early_stopping: Optional[int] = None, val_split: float = 0.1, **xgbkwargs
    ) -> None:
        print(f"Training {self.model_name} for experiment {self.experiment}")

        if early_stopping is not None:
            self.early_stopping = True
            len_mask = len(
                DataLoader._load_datasets(
                    self.data_path,
                    mode="train",
                    shuffle_data=False,
                    experiment=self.experiment,
                )
            )
            train_mask, val_mask = train_val_mask(len_mask, val_split)

            train_dataloader = self.get_dataloader(
                mode="train", mask=train_mask, shuffle_data=True
            )
            val_dataloader = self.get_dataloader(
                mode="train", mask=val_mask, shuffle_data=False
            )

        else:
            train_dataloader = self.get_dataloader(mode="train", shuffle_data=True)

        if "objective" not in xgbkwargs:
            xgbkwargs["objective"] = "reg:squarederror"
        self.model: xgb.XGBRegressor = xgb.XGBRegressor(**xgbkwargs)  # type: ignore

        # first, we need to collect all the data into arrays
        input_train_x, input_train_y = [], []

        for x, y in train_dataloader:
            input_train_x.append(self._concatenate_data(x))
            input_train_y.append(y)

        input_train_x_np = np.concatenate(input_train_x, axis=0)
        input_train_y_np = np.concatenate(input_train_y)

        fit_inputs = {"X": input_train_x_np, "y": input_train_y_np}

        if early_stopping is not None:
            input_val_x, input_val_y = [], []

            for val_x, val_y in val_dataloader:
                input_val_x.append(self._concatenate_data(val_x))
                input_val_y.append(val_y)

            input_val_x_np = np.concatenate(input_val_x, axis=0)
            input_val_y_np = np.concatenate(input_val_y)

            fit_val_inputs = {
                "eval_set": [input_val_x_np, input_val_y_np],
                "early_stopping_rounds": early_stopping,
                "eval_metric": "rmse",
            }
            fit_inputs.update(fit_val_inputs)

        self.model.fit(**fit_inputs)  # type: ignore

    def explain(
        self, x: Optional[TrainData] = None, save_shap_values: bool = True
    ) -> np.ndarray:

        assert self.model is not None, "Model must be trained!"

        if x is None:
            test_arrays_loader = self.get_dataloader(
                mode="test", shuffle_data=False, batch_file_size=1
            )
            _, val = list(next(iter(test_arrays_loader)).items())[0]
            x = val.x

        reshaped_x = self._concatenate_data(x)

        pred_x = xgb.DMatrix(reshaped_x)  # type: ignore

        input_dict = {"data": pred_x, "pred_contribs": True, "validate_features": False}

        if self.early_stopping:
            input_dict["ntree_limt"] = self.model.best_ntree_limit

        explanations = self.model.get_booster().predict(**input_dict)

        if save_shap_values:
            analysis_folder = self.model_dir / "analysis"
            if not analysis_folder.exists():
                analysis_folder.mkdir()

            np.save(analysis_folder / f"shap_values.npy", explanations)
            np.save(analysis_folder / f"input.npy", reshaped_x)

        return explanations

    def save_model(self) -> None:

        assert self.model is not None, "Model must be trained!"

        model_data = {
            "model": {"model": self.model, "early_stopping": self.early_stopping},
            "experiment": self.experiment,
            "pred_months": self.pred_months,
            "include_pred_month": self.include_pred_month,
            "surrounding_pixels": self.surrounding_pixels,
            "batch_size": self.batch_size,
            "ignore_vars": self.ignore_vars,
            "include_monthly_aggs": self.include_monthly_aggs,
            "include_yearly_aggs": self.include_yearly_aggs,
            "static": self.static,
            "early_stopping": self.early_stopping,
            "spatial_mask": self.spatial_mask,
            "include_prev_y": self.include_prev_y,
            "normalize_y": self.normalize_y,
        }

        with (self.model_dir / "model.pkl").open("wb") as f:
            pickle.dump(model_data, f)

    def load(self, model: Any, early_stopping: bool) -> None:
        assert isinstance(model, xgb.XGBRegressor)  # type: ignore
        self.model = model
        self.early_stopping = early_stopping

    def predict(self) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, np.ndarray]]:
        test_arrays_loader = self.get_dataloader(mode="test", shuffle_data=False)

        preds_dict: Dict[str, np.ndarray] = {}
        test_arrays_dict: Dict[str, Dict[str, np.ndarray]] = {}

        assert self.model is not None, "Model must be trained!"

        for dict in test_arrays_loader:
            for key, val in dict.items():
                x = self._concatenate_data(val.x)
                preds = self.model.predict(x)
                preds_dict[key] = preds
                test_arrays_dict[key] = {
                    "y": val.y,
                    "latlons": val.latlons,
                    "time": val.target_time,
                    "y_var": val.y_var,
                }
                if self.predict_delta:
                    assert val.historical_target.shape[0] == val.y.shape[0], (
                        "Expect"
                        f"the shape of the y ({val.y.shape})"
                        f" and historical_target ({val.historical_target.shape})"
                        " to be the same!"
                    )
                    test_arrays_dict[key]["historical_target"] = val.historical_target

        return test_arrays_dict, preds_dict
