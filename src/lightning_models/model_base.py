from argparse import Namespace
import pytorch_lightning as pl

from src.models.data import DataLoader, train_val_mask
from src.models.dynamic_data import DynamicDataLoader

from typing import cast, Any, Dict, Optional, Union, List


class LightningBase(pl.LightningModule):
    def __init__(self, hparams: Namespace) -> None:
        super().__init__()
        self.hparams = hparams
        self.data_folder = hparams.data_folder

        # needs to be set by the train function
        self.num_locations: Optional[int] = None

        # lots of attributes added by the dataloader
        dataloader = self.get_dataloader(mode="train", shuffle=False)
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
        )

    def val_dataloader(self):
        return self.get_dataloader(mode="train", shuffle_data=False, mask=self.val_mask)

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
                "data_path": self.data_path,
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
                "data_path": self.data_path,
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
