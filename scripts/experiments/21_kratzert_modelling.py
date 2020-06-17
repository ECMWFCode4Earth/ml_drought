import sys
from pathlib import Path

sys.path.append("../..")

from src.preprocess.camels_kratzert import (
    get_basins,
    RunoffEngineer,
    CamelsH5,
)
from src.preprocess import CAMELSGBPreprocessor

import pandas as pd
import numpy as np
import h5py
import pytest
from torch.utils.data import DataLoader

from src.models.kratzert.main import train as train_model
from src.models.kratzert.main import evaluate as evaluate_model

from scripts.utils import (
    _rename_directory,
    get_data_path,
    rename_features_dir,
    rename_models_dir,
)


def preprocess(data_dir: Path):
    processor = CAMELSGBPreprocessor(data_dir, open_shapefile=False)
    processor.preprocess()


def engineer(data_dir: Path):
    engineer = RunoffEngineer()
    engineer.create_training_data()


def run_model(**kwargs):
    model = train_model(**kwargs)

    eval_model_kwargs = dict(
        input_size_dyn=model.input_size_dyn,
        input_size_stat=model.input_size_stat,
        model_path = model.model_path
    )

    updated_kwargs = {**kwargs, **eval_model_kwargs}
    evaluate_model(
        **updated_kwargs
    )


def __main__():
    data_dir = get_data_path()
    preprocess(data_dir)

    # SETTINGS
    all_settings = dict(
        data_dir=data_dir,
        train_dates=[2000],
        val_dates=[2001]
        target_var="discharge_spec",
        x_variables=["precipitation", "peti"],
        static_variables=["pet_mean", "aridity", "p_seasonality"],
        seq_length=10,
        basins=get_basins(data_dir),
        dropout=0.4,
        hidden_size=256,
        seed=10101,
        cache=True,
        use_mse=True,
        batch_size=50,
        num_workers=4,
        initial_forget_gate_bias=5,
        learning_rate=1e-3,
        epochs=30,
    )
    lstm_settings = dict(
        with_static=True,
        concat_static=True,
    )
    ealstm_settings = dict(
        with_static=True,
        concat_static=False,
    )

    # run ealstm
    kwargs = {**all_settings, **ealstm_settings}
    run_model(**kwargs)

    # run lstm
    kwargs = {**all_settings, **lstm_settings}
    run_model(**kwargs)



