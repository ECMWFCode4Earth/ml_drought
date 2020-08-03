import numpy as np
from pathlib import Path
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import pickle
import xarray as xr

from typing import cast, Dict, List, Tuple, Optional, Union

from .base import ModelBase
from .utils import chunk_array
from .data import DataLoader, train_val_mask, TrainData

shap = None


class LinearRegression(ModelBase):

    model_name = "linear_regression"

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
        explain: bool = False,
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
        if explain:
            global shap
            if shap is None:
                import shap
            self.explainer: Optional[shap.DeepExplainer] = None  # type: ignore

    def train(
        self,
        num_epochs: int = 1,
        early_stopping: Optional[int] = None,
        batch_size: int = 256,
        val_split: float = 0.1,
        initial_learning_rate: float = 1e-15,
    ) -> None:
        print(f"Training {self.model_name} for experiment {self.experiment}")

        if early_stopping is not None:
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

            batches_without_improvement = 0
            best_val_score = np.inf
        else:
            train_dataloader = self.get_dataloader(mode="train", shuffle_data=True)

        self.model: linear_model.SGDRegressor = linear_model.SGDRegressor(
            eta0=initial_learning_rate
        )

        for epoch in range(num_epochs):
            train_rmse = []
            for x, y in train_dataloader:
                for batch_x, batch_y in chunk_array(x, y, batch_size, shuffle=True):
                    batch_y = cast(np.ndarray, batch_y)
                    x_in = self._concatenate_data(batch_x)

                    if x_in.shape[0] == 0:
                        pass

                    # fit the model
                    self.model.partial_fit(x_in, batch_y.ravel())
                    # evaluate the fit
                    train_pred_y = self.model.predict(x_in)
                    train_rmse.append(
                        np.sqrt(mean_squared_error(batch_y, train_pred_y))
                    )
            if early_stopping is not None:
                val_rmse = []
                for x, y in val_dataloader:
                    x_in = self._concatenate_data(x)
                    val_pred_y = self.model.predict(x_in)
                    val_rmse.append(np.sqrt(mean_squared_error(y, val_pred_y)))

            print(f"Epoch {epoch + 1}, train RMSE: {np.mean(train_rmse):.2f}")

            if early_stopping is not None:
                epoch_val_rmse = np.mean(val_rmse)
                print(f"Val RMSE: {epoch_val_rmse}")
                if epoch_val_rmse < best_val_score:
                    batches_without_improvement = 0
                    best_val_score = epoch_val_rmse
                    best_coef = self.model.coef_
                    best_intercept = self.model.intercept_
                else:
                    batches_without_improvement += 1
                    if batches_without_improvement == early_stopping:
                        print("Early stopping!")
                        self.model.coef_ = best_coef
                        self.model.intercept_ = best_intercept
                        return None

    def explain(
        self, x: Optional[TrainData] = None, save_shap_values: bool = True
    ) -> np.ndarray:

        assert self.model is not None, "Model must be trained!"

        if self.explainer is None:
            mean = self._calculate_big_mean()
            self.explainer: shap.LinearExplainer = shap.LinearExplainer(  # type: ignore
                self.model, (mean, None), feature_dependence="independent"
            )

        if x is None:
            test_arrays_loader = self.get_dataloader(
                mode="test", batch_file_size=1, shuffle_data=False
            )

            _, val = list(next(iter(test_arrays_loader)).items())[0]
            x = val.x

        reshaped_x = self._concatenate_data(x)
        explanations = self.explainer.shap_values(reshaped_x)  # type: ignore

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
            "model": {"coef": self.model.coef_, "intercept": self.model.intercept_},
            "experiment": self.experiment,
            "pred_months": self.pred_months,
            "include_pred_month": self.include_pred_month,
            "surrounding_pixels": self.surrounding_pixels,
            "batch_size": self.batch_size,
            "ignore_vars": self.ignore_vars,
            "include_monthly_aggs": self.include_monthly_aggs,
            "include_yearly_aggs": self.include_yearly_aggs,
            "static": self.static,
            "spatial_mask": self.spatial_mask,
            "include_prev_y": self.include_prev_y,
            "normalize_y": self.normalize_y,
        }

        with (self.model_dir / "model.pkl").open("wb") as f:
            pickle.dump(model_data, f)

    def load(self, coef: np.ndarray, intercept: np.ndarray) -> None:
        self.model: linear_model.SGDRegressor = linear_model.SGDRegressor()  # type: ignore
        self.model.coef_ = coef
        self.model.intercept_ = intercept

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

    def _calculate_big_mean(self) -> np.ndarray:
        """
        Calculate the mean of the training data in batches.
        For now, we don't calculate the covariance matrix,
        since it wouldn't fit in memory either
        """
        print("Calculating the mean of the training data")
        train_dataloader = self.get_dataloader(
            mode="train", batch_file_size=1, shuffle_data=False
        )

        means, sizes = [], []
        for x, _ in train_dataloader:
            x_in = self._concatenate_data(x)
            sizes.append(x_in.shape[0])
            means.append(x_in.mean(axis=0))

        total_size = sum(sizes)
        weighted_means = [mean * size / total_size for mean, size in zip(means, sizes)]
        return sum(weighted_means)
