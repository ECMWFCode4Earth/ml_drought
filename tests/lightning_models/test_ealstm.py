import numpy as np
import pandas as pd
import pytest

from src.lightning_models.ealstm import EARecurrentNetwork


class TestLightningBase:
    def test_(self, tmp_path):
        l = EARecurrentNetwork(
            data_dir=tmp_path,
            experiment="one_month_forecast",
            include_pred_month=True,
            surrounding_pixels=None,
            pretrained=False,
            explain=False,
            static="features",
            ignore_vars=always_ignore_vars,
            num_epochs=num_epochs,
            early_stopping=early_stopping,
            hidden_size=hidden_size,
            static_embedding_size=static_size,
            include_latlons=True,
            include_yearly_aggs=False,
            clear_nans=True,
            weight_observations=False,
            pred_month_static=False,
        )
