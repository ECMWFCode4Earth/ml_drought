import torch
from torch import nn
import pytorch_lightning as pl

from pathlib import Path
from copy import copy
import xarray as xr

from typing import Dict, List, Optional, Tuple, Union

from .model_base import LightningBase


class EARecurrentNetwork(LightningBase):

    model_name = "ealstm"

    def __init__(
        self,
        hidden_size: int,
        dense_features: Optional[List[int]] = None,
        rnn_dropout: float = 0.25,
        data_folder: Path = Path("data"),
        batch_size: int = 1,
        experiment: str = "one_month_forecast",
        pred_months: Optional[List[int]] = None,
        include_latlons: bool = False,
        include_pred_month: bool = True,
        include_monthly_aggs: bool = True,
        include_yearly_aggs: bool = True,
        surrounding_pixels: Optional[int] = None,
        ignore_vars: Optional[List[str]] = None,
        static: Optional[str] = "features",
        static_embedding_size: Optional[int] = None,
        device: str = "cuda:0",
        predict_delta: bool = False,
        spatial_mask: Union[xr.DataArray, Path] = None,
        include_prev_y: bool = True,
        normalize_y: bool = True,
    ):
        super(EARecurrentNetwork, self).__init__(
            data_folder=data_folder,
            batch_size=batch_size,
            experiment=experiment,
            pred_months=pred_months,
            include_pred_month=include_pred_month,
            include_latlons=include_latlons,
            include_monthly_aggs=include_monthly_aggs,
            include_yearly_aggs=include_yearly_aggs,
            surrounding_pixels=surrounding_pixels,
            ignore_vars=ignore_vars,
            static=static,
            device=device,
            predict_delta=predict_delta,
            spatial_mask=spatial_mask,
            include_prev_y=include_prev_y,
            normalize_y=normalize_y,
        )

        #

    def forward(
        self,
        x,
        pred_month=None,
        latlons=None,
        current=None,
        yearly_aggs=None,
        static=None,
        prev_y=None,
    ):
        """Forward pass through the entire network.
        Through an input_layer -> recurrent layer -> fully_connected_layer
        Returns a scalar prediction.
        """

        assert (
            (yearly_aggs is not None) or (latlons is not None) or (static is not None)
        ), "latlons, yearly means and static can't all be None"

        static_x = []
        if self.include_latlons:
            assert latlons is not None
            static_x.append(latlons)
        if self.include_yearly_agg:
            assert yearly_aggs is not None
            static_x.append(yearly_aggs)
        if self.include_static:
            static_x.append(static)
        if self.include_pred_month:
            static_x.append(pred_month)
        if self.include_prev_y:
            static_x.append(prev_y)

        static_tensor = torch.cat(static_x, dim=-1)

        if self.use_static_embedding:
            static_tensor = self.static_embedding(static_tensor)

        hidden_state, cell_state = self.rnn(x, static_tensor)

        x = self.dropout(hidden_state[:, -1, :])

        if self.experiment == "nowcast":
            assert current is not None
            x = torch.cat((x, current), dim=-1)

        for layer_number, dense_layer in enumerate(self.dense_layers):
            x = dense_layer(x)
        return x

