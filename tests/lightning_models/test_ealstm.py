import numpy as np
import pandas as pd
import random
import pytest
from argparse import Namespace
import pytorch_lightning as pl

from src.lightning_models.ealstm import EALSTM
from src.lightning_models.model_base import LightningModel
from tests.utils import make_drought_test_data, get_dataloader


class TestLightningBase:
    def test_initialize(self, tmp_path):
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
                "val_ratio": 0.3,
                "learning_rate": 1e3,
            }
        )

        dataloader = get_dataloader(mode="train", hparams=hparams, shuffle_data=False)
        x_ref, _ = next(iter(dataloader))
        EALSTM._initialize_model(x_ref=x_ref, hparams=hparams)

    def test_train(self, tmp_path, capsys):
        random.seed(1)
        np.random.seed(1)
        x, y, static = make_drought_test_data(tmp_path, len_dates=4, test=False)
        x_test, y_test, _ = make_drought_test_data(tmp_path, len_dates=2, test=True)
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
            }
        )

        model = LightningModel(hparams)

        # print the output to the console
        with capsys.disabled():
            kwargs = dict(fast_dev_run=True)
            model.fit(**kwargs)

            kwargs = dict()
            model.predict()
