import numpy as np
import pandas as pd
import pytest

from src.lightning_models.model_base import LightningBase

from ..utils import _make_dataset, _create_runoff_features_dir
import pytorch_lightning as pl


class TestLightningBase:
    def test_(self, tmp_path):
        DYNAMIC = True

        ds, static = _create_runoff_features_dir(tmp_path)

        static_ignore_vars = ["area"]
        dynamic_ignore_vars = ["discharge"]
        target_var = "discharge"
        seq_length = 5
        test_years = [2001]
        forecast_horizon = 0
        batch_file_size = 1
        static_embedding_size = 5
        hidden_size = 5
        rnn_dropout = 0.3
        dropout = 0.3
        dense_features = None

        train_years = [1999]
        val_years = [2000]
        clip_values_to_zero = True

        loss_func = "MSE"
        learning_rate = 1e-4
        N_epochs = 10

        l = EARecurrentNetwork(
            dynamic=DYNAMIC,
            data_folder=tmp_path,
            experiment="one_timestep_forecast",
            dynamic_ignore_vars=dynamic_ignore_vars,
            static_ignore_vars=static_ignore_vars,
            target_var=target_var,
            seq_length=seq_length,
            test_years=test_years,
            forecast_horizon=forecast_horizon,
            batch_size=batch_file_size,
            static_embedding_size=static_embedding_size,
            hidden_size=hidden_size,
            rnn_dropout=rnn_dropout,
            dropout=dropout,
            dense_features=dense_features,
            include_latlons=False,
            include_pred_month=False,
            include_timestep_aggs=False,
            include_yearly_aggs=False,
            val_years=val_years,
            train_years=train_years,
            clip_values_to_zero=clip_values_to_zero,
        )

    trainer = pl.Trainer(max_epochs=N_epochs)
