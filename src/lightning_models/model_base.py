from argparse import Namespace
import pytorch_lightning as pl
import torch
from torch.nn import functional as F

from .dataloader import DataLoader

from typing import cast, Any, Dict, Optional, Union, Tuple
from .ealstm import EALSTM
from .dataset import TrainData, train_val_mask
from src.models.utils import chunk_array
# from .lstm import LSTM
# from .linear_nn import LinearNN


class LightningModel(pl.LightningModule):
    def __init__(self, hparams: Namespace) -> None:
        super().__init__()
        self.hparams = hparams
        self.data_path = hparams.data_path

        # needs to be set by the train function
        self.num_locations: Optional[int] = None

        # lots of attributes added by the dataloader
        dataloader = self.get_dataloader(mode="train", shuffle_data=True, mask=None)
        num_examples = len(dataloader)
        x_ref, _ = next(iter(dataloader))

        # initialize the model
        self.model = self._initialize_model(x_ref, hparams)

        # train validation split
        self.train_mask, self.val_mask = train_val_mask(
            num_examples, self.hparams.val_ratio
        )

    def configure_optimizers(self):
        return torch.optim.Adam(
            [pam for pam in self.model.parameters()], lr=self.hparams.learning_rate
        )

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def train_dataloader(self):
        return self.get_dataloader(
            mode="train", shuffle_data=True, mask=self.train_mask
        )

    def val_dataloader(self):
        return self.get_dataloader(mode="train", shuffle_data=False, mask=self.val_mask)

    def training_step(self, batch, batch_idx):
        x, y = batch
        for x_batch, y_batch in chunk_array(x, y, self.hparams.batch_size, shuffle=True):
            pred = self.model(
                *self._input_to_tuple(cast(Tuple[torch.Tensor, ...], x_batch))
            )
            loss = F.smooth_l1_loss(pred, y_batch)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        for x_batch, y_batch in chunk_array(x, y, self.hparams.batch_size, shuffle=False):
            pred = self.model(
                *self._input_to_tuple(cast(Tuple[torch.Tensor, ...], x_batch))
            )
            loss = F.smooth_l1_loss(pred, y_batch)
        return {"val_loss": loss}

    # def validation_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

    #     tensorboard_logs = {"val_loss": avg_loss}

    #     return {"val_loss": avg_loss, "log": tensorboard_logs}

    def test_dataloader(self):
        return self.get_dataloader(
            mode="test", shuffle_data=True,
        )

    def test_step(self, batch, batch_idx):
        x, y = batch
        for x_batch, y_batch in chunk_array(x, y, self.hparams.batch_size, shuffle=False):
            pred = self.model(
                *self._input_to_tuple(cast(Tuple[torch.Tensor, ...], x_batch))
            )
            loss = F.smooth_l1_loss(pred, y_batch)

        return {"test_loss": loss}


    # ------------------------------------------------------
    # CUSTOM METHODS
    # ------------------------------------------------------
    @staticmethod
    def _initialize_model(x_ref: Tuple[torch.Tensor, ...], hparams: Namespace) -> torch.nn.Module:
        model_lookup = {
            "ealstm": EALSTM,
            # "lstm": LSTM,
            # "dense": LinearNN,
        }
        model_name = hparams.model_name.lower()
        assert (
            model_name in [k for k in model_lookup.keys()]
        ), f"Model not found in: {[k for k in model_lookup.keys()]}"
        model_class = model_lookup[model_name]

        # initialize the child_class
        model = model_class._initialize_model(x_ref, hparams)

        return model

    def _input_to_tuple(
        self, x: Union[Tuple[torch.Tensor, ...], TrainData],
    ) -> Tuple[torch.Tensor, ...]:
        """
        Returns:
        --------
        Tuple:
            [0] historical data
            [1] months (one hot encoded)
            [2] latlons
            [3] current data
            [4] yearly aggregations
            [5] static data
            [6] prev y var
        """
        # mypy totally fails to handle what's going on here

        if type(x) is TrainData:  # type: ignore
            return (  # type: ignore
                x.historical,  # type: ignore
                self._one_hot(x.pred_months, 12),  # type: ignore
                x.latlons,  # type: ignore
                x.current,  # type: ignore
                x.yearly_aggs,  # type: ignore
                self._one_hot(x.static, self.num_locations)  # type: ignore
                if self.hparams.static == "embeddings"
                else x.static,  # type: ignore
                x.prev_y_var,  # type: ignore
            )
        else:
            return (
                x[0],  # type: ignore
                self._one_hot(x[1], 12),  # type: ignore
                x[2],  # type: ignore
                x[3],  # type: ignore
                x[4],  # type: ignore
                self._one_hot(x[5], self.num_locations)  # type: ignore
                if self.hparams.static == "embeddings"
                else x[5],  # type: ignore
                x[6],  # type: ignore
            )

    def _one_hot(self, indices: torch.Tensor, num_vals: int) -> torch.Tensor:
        if len(indices.shape) > 1:
            indices = indices.squeeze(-1)
        return torch.eye(num_vals + 2, device=self.device)[indices.long()][:, 1:-1]

    def get_dataloader(
        self, mode: str, shuffle_data: bool = False, **kwargs
    ) -> DataLoader:
        """
        Return the correct dataloader for this model
        """

        default_args: Dict[str, Any] = {
            "data_path": self.hparams.data_path,
            "batch_file_size": self.hparams.batch_size,
            "shuffle_data": shuffle_data,
            "mode": mode,
            "mask": None,
            "experiment": self.hparams.experiment,
            "pred_months": self.hparams.pred_months,
            "to_tensor": True,
            "ignore_vars": self.hparams.ignore_vars,
            "monthly_aggs": self.hparams.include_monthly_aggs,
            "surrounding_pixels": self.hparams.surrounding_pixels,
            "static": self.hparams.static,
            "clear_nans": True,
            "normalize": True,
            "predict_delta": self.hparams.predict_delta,
            "spatial_mask": self.hparams.spatial_mask,
            "normalize_y": self.hparams.normalize_y,
        }

        for key, val in kwargs.items():
            # override the default args
            default_args[key] = val

        dl = DataLoader(**default_args)

        if (self.hparams.static == "embeddings") and (self.num_locations is None):
            self.num_locations = cast(int, dl.max_loc_int)
        return dl
