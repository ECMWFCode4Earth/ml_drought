import torch
import numpy as np
import pytest
import xarray as xr
import pandas as pd

from src.models.data import DataLoader, _BaseIter, TrainData
from ..utils import _create_runoff_features_dir

class TestRunoffModelling:
    def test_dataloader(self, tmp_path):
        X_data, y_data, static_data = _create_runoff_features_dir(tmp_path)

        dl = DataLoader(tmp_path, mode='train', static=True)

