from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd
from pandas import Timestamp
from random import shuffle
from pathlib import Path
import pickle
import torch
import xarray as xr

from typing import cast, Dict, Optional, Union, List, Tuple


# The dict below maps the indices of the arrays returned
# by the training iterator to their name. It should be updated
# if new inputs are added
idx_to_input = {
    0: "historical",
    1: "pred_months",
    2: "latlons",
    3: "current",
    4: "yearly_aggs",
    5: "static",
    6: "prev_y_var",
}


@dataclass
class TrainData:
    historical: Union[np.ndarray, torch.Tensor]
    current: Union[np.ndarray, torch.Tensor, None]
    pred_months: Union[np.ndarray, torch.Tensor]
    # latlons are repeated here so they can be tensor-ized and
    # normalized
    latlons: Union[np.ndarray, torch.Tensor]
    yearly_aggs: Union[np.ndarray, torch.Tensor]
    static: Union[np.ndarray, torch.Tensor, None]
    prev_y_var: Union[np.ndarray, torch.Tensor]

    def to_tensor(self, device: torch.device) -> None:
        for key, val in self.__dict__.items():
            if val is not None:
                if type(val) is np.ndarray:
                    setattr(self, key, torch.from_numpy(val).to(device).float())

    def concatenate(self, x: TrainData):
        # Note this function will only concatenate values from x
        # which are not None in self
        for key, val in self.__dict__.items():
            if val is not None:
                if type(val) is np.ndarray:
                    newval = np.concatenate((val, getattr(x, key)), axis=0)
                else:
                    newval = torch.cat((val, getattr(x, key)), dim=0)
                setattr(self, key, newval)

    def filter(self, filter_array: Union[np.ndarray, torch.Tensor]) -> None:
        # noqa because we just want to make sure these to have the same type,
        # so isinstance doesn't really fit the bill
        assert type(filter_array) == type(  # noqa
            self.historical
        ), f"Got a different filter type from the TrainData arrays"
        for key, val in self.__dict__.items():
            if val is not None:
                setattr(self, key, val[filter_array])


@dataclass
class ModelArrays:
    x: TrainData
    y: Union[np.ndarray, torch.Tensor]
    x_vars: List[str]
    y_var: str
    latlons: Optional[np.ndarray] = None
    target_time: Optional[Timestamp] = None
    historical_times: Optional[List[Timestamp]] = None
    predict_delta: bool = False
    historical_target: Optional[xr.DataArray] = None

    def to_tensor(self, device) -> None:
        self.x.to_tensor(device)
        if type(self.y) is np.ndarray:
            self.y = torch.from_numpy(self.y).to(device).float()

    def concatenate(self, x: ModelArrays) -> None:
        self.x.concatenate(x.x)

        if type(self.y) is np.ndarray:
            self.y = np.concatenate((self.y, x.y), axis=0)
        else:
            self.y = torch.cat((self.y, x.y), dim=0)

        if self.latlons is not None:
            self.latlons = np.concatenate((self.latlons, x.latlons), axis=0)

    def to_xarray(self) -> Tuple[xr.Dataset, xr.Dataset, Optional[xr.Dataset]]:
        assert (
            self.latlons.shape[0] == self.x.historical.shape[0]  # type: ignore
        ), "first dim is # pixels"
        assert (
            len(self.x_vars) == self.x.historical.shape[2]
        ), "final dim is # variables"

        # if experiment == 'nowcast'
        if self.x.current is not None:
            # NOTE: historical times is one less for nowcast?
            assert (
                len(self.historical_times) == self.x.historical.shape[1]  # type: ignore
            ), "second dim is # timesteps"
        else:
            assert (
                len(self.historical_times) == self.x.historical.shape[1]  # type: ignore
            ), "second dim is # timesteps"
        variables = self.x_vars
        latitudes = np.unique(self.latlons[:, 0])  # type: ignore
        longitudes = np.unique(self.latlons[:, 1])  # type: ignore
        times = self.historical_times

        ds_list = []
        for i, variable in enumerate(variables):
            # for each variable create the indexed Dataset
            # from the x.historical array
            data_np = self.x.historical[:, :, i]
            ds_list.append(
                xr.Dataset(
                    {
                        variable: (
                            ["lat", "lon", "time"],
                            # unflatten the (pixel, time) array -> (lat, lon, time)
                            data_np.reshape(
                                len(latitudes),
                                len(longitudes),
                                len(times),  # type: ignore
                            ),
                        )
                    },
                    coords={"lat": latitudes, "lon": longitudes, "time": times},
                )
            )

        # TODO: create the static Dataset too!
        historical_ds = xr.auto_combine(ds_list)

        # create the target_ds
        target_ds = xr.Dataset(
            {
                self.y_var: (
                    ["lat", "lon", "time"],
                    self.y.reshape(len(latitudes), len(longitudes), 1),
                )
            },
            coords={"lat": latitudes, "lon": longitudes, "time": [self.target_time]},
        )

        # if experiment == 'nowcast'
        if self.x.current is not None:
            # get the current variables only
            current_vars = [
                (i - 1, v)
                for i, v in enumerate(self.x_vars)
                if (not v == self.y_var) & ("mean" not in v)
            ]
            current_ds_list = []
            for i, current_var in current_vars:
                current_ds_list.append(
                    xr.Dataset(
                        {
                            current_var: (
                                ["lat", "lon", "time"],
                                self.x.current[:, i].reshape(
                                    len(latitudes), len(longitudes), 1
                                ),
                            )
                        },
                        coords={
                            "lat": latitudes,
                            "lon": longitudes,
                            "time": [self.target_time],
                        },
                    )
                )
            current_ds = xr.auto_combine(current_ds_list)
        else:  # one_month_forecast
            current_ds = None

        return historical_ds, target_ds, current_ds


def train_val_mask(
    mask_len: int, val_ratio: float = 0.3
) -> Tuple[List[bool], List[bool]]:
    """Makes a training and validation mask which can be passed to the dataloader
    Arguments
    ----------
    mask_len: int
        The length of the mask to be created
    val_ratio: float = 0.3
        The ratio of instances which should be True for the val mask and False for the train
        mask
    Returns
    ----------
    The train mask and the val mask, both as lists
    """
    assert val_ratio < 1, f"Val ratio must be smaller than 1"
    train_mask = np.random.rand(mask_len) < 1 - val_ratio
    val_mask = ~train_mask

    return train_mask.tolist(), val_mask.tolist()
