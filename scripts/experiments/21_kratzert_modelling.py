import sys
from pathlib import Path

sys.path.append("../..")

from src.preprocess import CAMELSGBPreprocessor
from src.engineer.runoff import RunoffEngineer, CamelsDataLoader
from src.engineer.runoff_utils import get_basins

import pandas as pd
import numpy as np
import h5py
import pytest
from torch.utils.data import DataLoader

from src.models.kratzert.main import train
from src.models.kratzert.main import evaluate

from scripts.utils import (
    _rename_directory,
    get_data_path,
    rename_features_dir,
    rename_models_dir,
)
import inspect
from typing import Dict


def get_valid_kwargs(py_object, kwargs: Dict) -> Dict:
    """return a dict of the valid kwargs for a function
    """
    signature = inspect.signature(py_object)
    valid_kwargs = {}
    for arg in [k for k in signature.parameters.keys()]:
        valid_kwargs[arg] = kwargs[arg]

    return valid_kwargs


def preprocess(data_dir: Path):
    processor = CAMELSGBPreprocessor(data_dir, open_shapefile=False)
    processor.preprocess()
    print("**Data Preprocessed**")


def engineer(**kwargs):
    """Take the preprocessed data and produce X, y pairs
        data_dir: Path,
        basins: List[int],
        train_dates: List[int],
        with_basin_str: bool = True,
        target_var: str = "discharge_spec",
        x_variables: Optional[List[str]] = ["precipitation", "peti"],
        static_variables: Optional[List[str]] = None,
        ignore_static_vars: Optional[List[str]] = None,
        seq_length: int = 365,
        with_static: bool = True,
        concat_static: bool = False,
    """
    # only pass the valid kwargs from dict to engineer
    valid_kwargs = get_valid_kwargs(RunoffEngineer, kwargs)

    engineer = RunoffEngineer(**valid_kwargs)
    engineer.create_training_data()
    print("**Training Data Created (engineer)**")


def run_model(**kwargs):
    model = train_model(**kwargs)
    evaluate_model(model, **kwargs)


def train_model(**kwargs):
    # only pass the valid kwargs from dict to engineer
    valid_kwargs = get_valid_kwargs(train, kwargs)

    model = train(**valid_kwargs)
    print("**Model Trained**")
    return model


def evaluate_model(model, **kwargs):
    eval_model_kwargs = dict(
        input_size_dyn=model.input_size_dyn,
        input_size_stat=model.input_size_stat,
        model_path=model.model_path,
    )
    kwargs = {**kwargs, **eval_model_kwargs}
    valid_kwargs = get_valid_kwargs(evaluate, kwargs)

    evaluate(**valid_kwargs)
    print("**Model Evaluated**")


if __name__ == "__main__":
    data_dir = get_data_path()
    preprocess(data_dir)

    # SETTINGS
    all_settings = dict(
        data_dir=data_dir,
        basins=get_basins(data_dir),
        train_dates=[2000, 2010],
        with_basin_str=True,
        val_dates=[2011, 2020],
        target_var="discharge_spec",
        x_variables=["precipitation", "peti"],
        static_variables=["pet_mean", "aridity", "p_seasonality"],
        ignore_static_vars=None,
        seq_length=365,
        dropout=0.4,
        hidden_size=256,
        seed=10101,
        cache=True,
        use_mse=True,
        batch_size=1000,  # 50,
        num_workers=4,
        initial_forget_gate_bias=5,
        learning_rate=1e-3,
        epochs=10,
    )
    lstm_settings = dict(with_static=True, concat_static=True)
    ealstm_settings = dict(with_static=True, concat_static=False)

    # # run ealstm
    kwargs = {**all_settings, **ealstm_settings}
    run_model(**kwargs)

    # # run lstm
    # kwargs = {**all_settings, **lstm_settings}
    # run_model(**kwargs)

    # engineer data only
    # engineer(**kwargs)
