from pathlib import Path
import pytorch_lightning as pl
import xarray as xr
from torch.nn import functional as F
import torch

from src.models.data import DataLoader, train_val_mask, TrainData
from src.models.dynamic_data import DynamicDataLoader
from src.models.neural_networks.nseloss import NSELoss

from typing import cast, Any, Dict, Optional, Union, List, Tuple


class LightningBase(pl.LightningModule):
    def __init__(
        self,
        data_folder: Path = Path("data"),
        dynamic: bool = False,
        batch_size: int = 256,
        experiment: str = "one_month_forecast",
        forecast_horizon: int = 1,
        target_var: Optional[str] = None,
        test_years: Optional[Union[List[str], str]] = None,
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
        device: str = "cuda:0",
        predict_delta: bool = False,
        spatial_mask: Union[xr.DataArray, Path] = None,
        include_prev_y: bool = True,
        normalize_y: bool = True,
        val_years: Optional[List[Union[float, int]]] = None,
        train_years: Optional[List[Union[float, int]]] = None,
        clip_values_to_zero: bool = False,
    ) -> None:
        super().__init__()

        # attributes from the model base
        self.dynamic = dynamic
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.experiment = experiment
        self.seq_length = seq_length
        self.include_pred_month = include_pred_month
        self.include_latlons = include_latlons
        self.include_timestep_aggs = include_timestep_aggs
        self.include_yearly_aggs = include_yearly_aggs
        self.surrounding_pixels = surrounding_pixels
        self.ignore_vars = ignore_vars
        self.static = static
        self.predict_delta = predict_delta
        self.spatial_mask = spatial_mask
        self.include_prev_y = include_prev_y
        self.normalize_y = normalize_y
        self.pred_months = pred_months
        self.dynamic_ignore_vars = dynamic_ignore_vars
        self.static_ignore_vars = static_ignore_vars
        self.target_var = target_var
        self.test_years = test_years
        self.forecast_horizon = forecast_horizon
        self.clip_values_to_zero = clip_values_to_zero

        # needs to be set by the train function
        self.num_locations: Optional[int] = None

        # lots of attributes added by the dataloader
        dataloader = self.get_dataloader(mode="train")
        num_examples = len(dataloader)
        x_ref, _ = next(iter(dataloader))
        self.model = self._initialize_model(x_ref)

        self.train_mask, self.val_mask = train_val_mask(
            num_examples, self.hparams.val_ratio
        )

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def train_dataloader(self):
        return self.get_dataloader(
            mode="train", shuffle_data=True, mask=self.train_mask
            mode="train", shuffle_data=True, mask=self.train_mask
        )

    def training_step(self, batch, batch_idx):
        # ------- FORWARD PASS ---------
        pred = self.model(
            *self._input_to_tuple(cast(Tuple[torch.Tensor, ...], x_batch))
        )
        # ------- LOSS FUNCTION ---------
        if loss_func == "NSE":
            # NSELoss needs std of each basin for each sample
            if x_batch[7] is not None:
                target_var_std = x_batch[7]
                # (train_dataloader.__iter__()).target_var_std
            else:
                assert (
                    False
                ), "x[7] should not be None, this is the target_var_std"

            loss = NSELoss().forward(pred, y_batch, target_var_std)
        elif loss_func == "MSE":
            loss = F.mse_loss(pred, y_batch)
        elif loss_func == "huber":
            loss = F.smooth_l1_loss(pred, y_batch)
        else:
            assert False, "Only implemented MSE / NSE / huber loss functions"

        # ------- BACKWARD PASS ---------
        loss.backward()
        self.optimizer.step()

        # evaluation / keeping track of losses over time
        with torch.no_grad():
            mse = F.mse_loss(pred, y_batch)

        epoch_losses.append(loss.cpu().item())
        epoch_rmses.append(np.sqrt(mse.cpu().item()))

        assert len(epoch_losses) >= 1
        assert len(epoch_rmses) >= 1

        # TODO: check that most recent loss is notnan
        assert not np.isnan(epoch_losses[-1])
        return

    def val_dataloader(self):
        return self.get_dataloader(mode="train", shuffle_data=False, mask=self.val_mask)


    # --------- CUSTOM FUNCTIONS --------- #

    def _input_to_tuple(
        self, x: Union[Tuple[torch.Tensor, ...], TrainData]
    ) -> Tuple[Union[torch.Tensor, Any], ...]:
        """
        Returns:
        --------
        Tuple:
            [0] historical data
            [1] months (one hot encoded)
            [2] latlons (optional)
            [3] current data (optional - Nowcast)
            [4] yearly aggregations
            [5] static data (optional)
            [6] prev y var (optional)
            [7] y var std (optional - req for NSE loss function)
        """
        # mypy totally fails to handle what's going on here

        if type(x) is TrainData:  # type: ignore
            return (  # type: ignore
                x.historical,  # type: ignore
                self._one_hot(x.pred_month, 12)  # type: ignore
                if x.pred_month is not None  # type: ignore
                else None,
                x.latlons,  # type: ignore
                x.current,  # type: ignore
                x.yearly_aggs,  # type: ignore
                self._one_hot(x.static, self.num_locations)  # type: ignore
                if self.static == "embeddings"
                else x.static,  # type: ignore
                x.prev_y_var,  # type: ignore
            )
        else:
            return (
                x[0],  # type: ignore
                self._one_hot(x[1], 12) if x[1] is not None else None,  # type: ignore
                x[2],  # type: ignore
                x[3],  # type: ignore
                x[4],  # type: ignore
                self._one_hot(x[5], self.num_locations)  # type: ignore
                if self.static == "embeddings"
                else x[5],  # type: ignore
                x[6],  # type: ignore
            )

    def get_dataloader(
        self,
        mode: str,
        to_tensor: bool = False,
        shuffle_data: bool = False,
        batch_file_size: int = 3,
        **kwargs,
    ) -> Union[DataLoader, DynamicDataLoader]:
        """
        Return the correct dataloader for this model
        """

        if self.dynamic:
            default_args: Dict[str, Any] = {
                "target_var": self.target_var,
                "test_years": self.test_years,
                "seq_length": self.seq_length,  # changed this default arg
                "forecast_horizon": self.forecast_horizon,
                "data_path": self.data_folder,
                "batch_file_size": batch_file_size,  #   self.batch_file_size,
                "mode": mode,
                "shuffle_data": True,
                "clear_nans": True,
                "normalize": True,
                "predict_delta": False,
                "experiment": "one_timestep_forecast",  # changed this default arg
                "mask": None,
                "pred_months": None,  # changed this default arg
                "to_tensor": to_tensor,
                "surrounding_pixels": False,
                "dynamic_ignore_vars": self.dynamic_ignore_vars,
                "static_ignore_vars": self.static_ignore_vars,
                "timestep_aggs": False,  # changed this default arg
                "static": self.static,
                "device": self.device,
                "spatial_mask": None,
                "normalize_y": True,
                "reducing_dims": None,
                "calculate_latlons": False,  # changed this default arg
                "use_prev_y_var": False,
                "resolution": "D",
            }

        else:
            default_args = {
                "data_path": self.data_folder,
                "batch_file_size": batch_file_size,
                "shuffle_data": shuffle_data,
                "mode": mode,
                "mask": None,
                "experiment": self.experiment,
                "seq_length": self.seq_length,
                "pred_months": self.pred_months,
                "to_tensor": to_tensor,
                "ignore_vars": self.ignore_vars,
                "timestep_aggs": self.include_timestep_aggs,
                "surrounding_pixels": self.surrounding_pixels,
                "static": self.static,
                "device": self.device,
                "clear_nans": True,
                "normalize": True,
                "predict_delta": self.predict_delta,
                "spatial_mask": self.spatial_mask,
                "normalize_y": self.normalize_y,
                "calculate_latlons": self.include_latlons,
                "use_prev_y_var": self.include_prev_y,
            }

        for key, val in kwargs.items():
            # override the default args
            default_args[key] = val

        dl: Union[DataLoader, DynamicDataLoader]
        if self.dynamic:
            dl = DynamicDataLoader(**default_args)
        else:
            dl = DataLoader(**default_args)

        if (self.static == "embeddings") and (self.num_locations is None):
            self.num_locations = cast(int, dl.max_loc_int)
        return dl
