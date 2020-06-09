import torch
from torch import nn
import pytorch_lightning as pl

from pathlib import Path
from copy import copy
import xarray as xr

from typing import Dict, List, Optional, Tuple, Union

from .base import NNBase


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