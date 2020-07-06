from pathlib import Path
import pytorch_lightning as pl
import xarray as xr
from torch.nn import functional as F
import torch

from copy import copy

from src.models.data import DataLoader, train_val_mask, TrainData
from src.models.dynamic_data import DynamicDataLoader
from src.models.neural_networks.nseloss import NSELoss

from typing import cast, Any, Dict, Optional, Union, List, Tuple


class LSTM(pl.LightningModule):
    def __init__(
        self,
        hidden_size: int,
        dense_features: Optional[List[int]] = None,
        dynamic: bool = False,
        rnn_dropout: float = 0.25,
        dropout: float = 0.25,
        data_folder: Path = Path("data"),
        forecast_horizon: int = 1,
        target_var: Optional[str] = None,
        test_years: Optional[Union[List[str], str]] = None,
        batch_size: int = 1,
        experiment: str = "one_month_forecast",
        seq_length: int = 365,
        pred_months: Optional[List[int]] = None,
        include_pred_month: bool = True,
        include_latlons: bool = False,
        include_timestep_aggs: bool = True,
        include_yearly_aggs: bool = True,
        surrounding_pixels: Optional[int] = None,
        ignore_vars: Optional[List[str]] = None,
        dynamic_ignore_vars: Optional[List[str]] = None,
        static_ignore_vars: Optional[List[str]] = None,
        static: Optional[str] = "features",
        predict_delta: bool = False,
        spatial_mask: Union[xr.DataArray, Path] = None,
        include_prev_y: bool = True,
        normalize_y: bool = True,
        val_years: Optional[List[Union[float, int]]] = None,
        train_years: Optional[List[Union[float, int]]] = None,
        clip_values_to_zero: bool = False,
    ):
        # base model code
        self.batch_size = batch_size
        self.include_pred_month = include_pred_month
        self.include_latlons = include_latlons
        self.include_timestep_aggs = include_timestep_aggs
        self.include_yearly_aggs = include_yearly_aggs
        self.data_path = data_folder
        self.experiment = experiment
        self.seq_length = seq_length
        self.pred_months = pred_months
        self.models_dir = data_folder / "models" / self.experiment
        self.surrounding_pixels = surrounding_pixels
        self.ignore_vars = ignore_vars
        self.dynamic_ignore_vars = dynamic_ignore_vars
        self.static_ignore_vars = static_ignore_vars
        self.static = static
        self.predict_delta = predict_delta
        self.include_prev_y = include_prev_y
        self.normalize_y = normalize_y
        if normalize_y:
            with (data_folder / f"features/{experiment}/normalizing_dict.pkl").open(
                "rb"
            ) as f:
                self.normalizing_dict = pickle.load(f)

        self.forecast_horizon = forecast_horizon
        self.dynamic = dynamic
        if self.dynamic:
            assert (
                target_var is not None
            ), "If using the dynamic DataLoader require a `target_var` parameter to be provided"
            assert (
                test_years is not None
            ), "If using the dynamic DataLoader require a `test_years` parameter to be provided"
            self.target_var = target_var
            self.test_years = test_years
            if self.include_yearly_aggs:
                print(
                    "`include_yearly_aggs` does not yet work for dynamic dataloder. Setting to False"
                )
                self.include_yearly_aggs = False
            if self.include_prev_y:
                print(
                    "`include_prev_y` does not yet work for dynamic dataloder. Setting to False"
                )
                self.include_prev_y = False

            print("Using the Dynamic DataLoader")
            print(f"\tTarget Var: {target_var}")
            print(f"\tTest Years: {test_years}")

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

        self.clip_values_to_zero = clip_values_to_zero

        # base NN code
        self.explainer: Optional[shap.DeepExplainer] = None

        self.train_years = train_years
        self.val_years = val_years

        if self.train_years is not None:
            assert not any(
                np.isin(test_years, train_years)
            ), "MODEL LEAKAGE - Train > Test"
        if self.val_years is not None:
            assert not any(np.isin(test_years, val_years)), "MODEL LEAKAGE - Val > Test"

        # LSTM code
        self.hidden_size = hidden_size
        self.rnn_dropout = rnn_dropout
        self.dropout = dropout
        self.input_dense_features = copy(dense_features)
        if dense_features is None:
            dense_features = []
        self.dense_features = dense_features
        self.target_var = target_var

        self.features_per_month: Optional[int] = None
        self.current_size: Optional[int] = None
        self.yearly_agg_size: Optional[int] = None
        self.static_size: Optional[int] = None


if __name__ == "__main__":
    # from argparse import ArgumentParser
    # parser = ArgumentParser()
    # # arguments to parse
    # parser = pl.Trainer.add_argparse_args(parser)

    model = LSTM()

    args = parser.parse_args()

    vae = VAE(hparams=args)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(vae)
