from argparse import Namespace
import pytorch_lightning as pl
import torch
import numpy as np
import pandas as pd
from torch.nn import functional as F
import xarray as xr
from pathlib import Path
from .dataloader import DataLoader

from typing import cast, Any, Dict, Optional, Union, Tuple, List
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
        self.model_name = hparams.model_name.lower()
        self.experiment = hparams.experiment

        self.model_dir: Path = self.data_path / "models" / self.experiment / self.model_name
        self.model_dir.mkdir(exist_ok=True, parents=True)

        # needs to be set by the train function
        self.num_locations: Optional[int] = None

        # lots of attributes added by the dataloader
        dataloader = self.get_dataloader(mode="train", shuffle_data=True, mask=None)
        num_examples = len(dataloader)
        assert False
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
        for x_batch, y_batch in chunk_array(
            x, y, self.hparams.batch_size, shuffle=True
        ):
            pred = self.model(
                *self._input_to_tuple(cast(Tuple[torch.Tensor, ...], x_batch))
            )
            loss = F.smooth_l1_loss(pred, y_batch)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        all_preds = []
        all_obs = []
        losses = []

        for x_batch, y_batch in chunk_array(
            x, y, self.hparams.batch_size, shuffle=False
        ):
            pred = self.model(
                *self._input_to_tuple(cast(Tuple[torch.Tensor, ...], x_batch))
            )
            loss = F.smooth_l1_loss(pred, y_batch)
            all_preds.append(pred)
            all_obs.append(y_batch)
            losses.append(loss)

        val_loss = torch.mean(torch.stack([l for l in losses]))
        all_preds = torch.stack([x for x in all_preds])
        all_obs = torch.stack([x for x in all_obs])

        return {"val_loss": loss, "preds": all_preds, "obs": all_obs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        preds = torch.stack([x["preds"] for x in outputs]).mean()
        obs = torch.stack([x["obs"] for x in outputs]).mean()

        mse = F.mse_loss(preds, obs)

        tensorboard_logs = {"val_loss": avg_loss, "val_mse": mse}

        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def test_dataloader(self):
        return self.get_dataloader(mode="test", shuffle_data=True)

    def test_step(self, batch, batch_idx) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]]]:
        # initilalise objects to write test results
        preds_dict: Dict[str, np.ndarray] = {}
        test_arrays_dict: Dict[str, Dict[str, np.ndarray]] = {}
        losses = []

        for key, val in batch.items():
            input_tuple = self._input_to_tuple(val.x)

            # TODO: generalise the chunk_array to work with
            # the test data as well as the train data
            num_sections = max(input_tuple[0].shape[0] // self.hparams.batch_size, 1)
            split_x = []
            for idx, x_section in enumerate(input_tuple):
                if x_section is not None:
                    split_x.append(torch.chunk(x_section, num_sections))
                else:
                    split_x.append([None] * num_sections)  # type: ignore

            chunked_input = list(zip(*split_x))

            all_preds = []
            for x_batch in chunked_input:
                # make test prediction
                pred = self.model(*x_batch)
                all_preds.append(pred)

            preds_dict[key] = np.concatenate(all_preds)

            test_arrays_dict[key] = {
                "y": val.y.numpy(),
                "latlons": val.latlons,
                "time": val.target_time,
                "y_var": val.y_var,
            }

        return preds_dict, test_arrays_dict

    def test_epoch_end(self, outputs: Tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]]]):
        # TODO: save the netcdf obs / sim files
        preds_dict: Dict[str, np.ndarray] = {}
        test_arrays_dict: Dict[str, Dict[str, np.ndarray]] = {}

        # recreate the bigger dictionary with all keys
        #  List[Dict[key, val]] -> Dict[keys, vals]
        for preds, test_arrays in outputs:
            key = [k for k in preds.keys()][0]
            value = preds[key]
            preds_dict[key] = value

            test_arrays_dict[key] = {}
            for key_l2 in [k for k in test_arrays[key].keys()]:
                test_arrays_dict[key][key_l2] = test_arrays[key][key_l2]

        if self.hparams.save_preds:
            for key, val in test_arrays_dict.items():
                latlons = cast(np.ndarray, val["latlons"])
                # preds = self.denormalize_y(preds_dict[key], val["y_var"])
                preds = preds_dict[key]
                # obs = self.denormalize_y(val["y"], val["y_var"])
                obs = val["y"]

                if len(preds.shape) > 1:
                    preds = preds.squeeze(-1)

                # the prediction timestep
                time = val["time"]
                times = [time for _ in range(len(preds))]

                preds_xr: xr.Dataset = (
                    pd.DataFrame(
                        data={
                            "preds": preds.flatten(),
                            "obs": obs.flatten(),
                            "lat": latlons[:, 0],
                            "lon": latlons[:, 1],
                            "time": times,
                        }
                    )
                    .set_index(["lat", "lon", "time"])
                    .to_xarray()
                )

                preds_xr.to_netcdf(self.model_dir / f"preds_{key}.nc")

        return

    def fit(self, **kwargs):
        if "weights_save_path" not in kwargs.keys():
            kwargs["weights_save_path"] = self.hparams.data_path

        trainer = pl.Trainer(**kwargs)
        trainer.fit(self)

    def predict(self, **kwargs):
        if "weights_save_path" not in kwargs.keys():
            kwargs["weights_save_path"] = self.hparams.data_path

        trainer = pl.Trainer(**kwargs)
        trainer.test(self)

    #  ------------------------------------------------------
    # CUSTOM METHODS
    #  ------------------------------------------------------
    @staticmethod
    def _initialize_model(
        x_ref: Tuple[torch.Tensor, ...], hparams: Namespace
    ) -> torch.nn.Module:
        model_lookup = {
            "ealstm": EALSTM,
            # "lstm": LSTM,
            # "dense": LinearNN,
        }
        model_name = hparams.model_name.lower()
        assert model_name in [
            k for k in model_lookup.keys()
        ], f"Model not found in: {[k for k in model_lookup.keys()]}"
        model_class = model_lookup[model_name]

        # initialize the child_class
        model = model_class._initialize_model(x_ref, hparams)

        return model

    def _input_to_tuple(
        self, x: Union[Tuple[torch.Tensor, ...], TrainData]
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
        return torch.eye(num_vals + 2)[indices.long()][:, 1:-1]

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
