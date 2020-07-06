# import numpy as np
# import pandas as pd
# import pytest
import pytorch_lightning as pl
from src.lightning_models.model_base import LightningBase
from tests.utils import _create_runoff_features_dir


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
        # static_embedding_size = 5
        # hidden_size = 5
        # rnn_dropout = 0.3
        # dropout = 0.3
        # dense_features = None

        train_years = [1999]
        val_years = [2000]
        clip_values_to_zero = True

        # loss_func = "MSE"
        # learning_rate = 1e-4
        # N_epochs = 10

        # ------ Instantiate model -----
        model = LightningBase(
            data_folder=tmp_path,
            dynamic=DYNAMIC,
            batch_size=batch_file_size,
            experiment="one_timestep_forecast",
            forecast_horizon=forecast_horizon,
            target_var=target_var,
            test_years=test_years,
            seq_length=seq_length,
            # optional extras
            pred_months=None,
            include_pred_month=False,
            include_latlons=False,
            include_timestep_aggs=False,
            include_yearly_aggs=False,
            surrounding_pixels=None,
            # ignore vars
            ignore_vars=None,
            dynamic_ignore_vars=dynamic_ignore_vars,
            static_ignore_vars=static_ignore_vars,
            static="features",
            val_years=val_years,
            train_years=train_years,
            clip_values_to_zero=clip_values_to_zero,
        )
        assert False
        assert isinstance(model, pl.LightningModule)

        # trainer = pl.Trainer(fast_dev_run=True)
