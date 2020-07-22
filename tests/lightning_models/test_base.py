import numpy as np
import pandas as pd
import pytest
from argparse import Namespace
from src.lightning_models.model_base import LightningModel
from tests.utils import make_drought_test_data, get_dataloader


class TestLightningModel:
    def test_(self, tmp_path):
        x, y, static = make_drought_test_data(tmp_path)
        hparams = Namespace(
            **{
                "model_name": "EALSTM",
                "data_path": tmp_path,
                "experiment": "one_month_forecast",
                "hidden_size": 64,
                "rnn_dropout": 0.3,
                "include_latlons": False,
                "static_embedding_size": 64,
                "include_prev_y": False,
                "include_yearly_aggs": False,
                "static": "features",
                "batch_size": 1,
                "include_pred_month": True,
                "pred_months": None,
                "ignore_vars": None,
                "include_monthly_aggs": False,
                "surrounding_pixels": None,
                "predict_delta": False,
                "spatial_mask": None,
                "normalize_y": True,
                "dense_features": [128],
                "val_ratio": 0.5,
                "learning_rate": 1e3,
                "save_preds": True,
            }
        )
        l = LightningModel(hparams)
