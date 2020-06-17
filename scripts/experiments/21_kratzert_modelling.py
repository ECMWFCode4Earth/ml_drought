import sys
from pathlib import Path

sys.path.append("../..")

from src.preprocess.camels_kratzert import (
    CalculateNormalizationParams,
    # reshape_data,
    CAMELSCSV,
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


def __main__():
    data_dir = get_data_path()

    preprocess(data_dir)
