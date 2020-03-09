from pathlib import Path
from typing import List, Tuple, Dict
import sys
import itertools

sys.path.append("../..")

from _base_models import parsimonious, regression, linear_nn, rnn, earnn

from scripts.utils import _rename_directory, get_data_path
from src.engineer.one_timestep_forecast import _OneTimestepForecastEngineer

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.utils import drop_nans_and_flatten
from src.analysis import read_train_data, read_test_data, read_pred_data

if __name__ == "__main__":
    data_dir = get_data_path()

    EXPERIMENT = "one_timestep_forecast"
    TRUE_EXPERIMENT = "one_timestep_forecast"
    TARGET_VAR = "discharge_vol"

    X_train, y_train = read_train_data(data_dir, experiment=TRUE_EXPERIMENT)
    X_test, y_test = read_test_data(data_dir, experiment=TRUE_EXPERIMENT)
    static_ds = xr.open_dataset(data_dir / "features/static/data.nc")

    assert False
    ds = xr.merge([y_train, y_test]).sortby("time").sortby("lat")
    d_ = xr.merge([X_train, X_test]).sortby("time").sortby("lat")
    ds = xr.merge([ds, d_])
