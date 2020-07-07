from argparse import Namespace
import pytorch_lightning as pl

from .data import DataLoader, train_val_mask

from typing import cast, Any, Dict, Optional, Union


class LightningBase(pl.LightningModule):
    def __init__(self, hparams: Union[Dict, Namespace]) -> None:
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
            "device": next(self.parameters()).device,
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
