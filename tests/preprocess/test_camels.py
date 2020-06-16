import xarray as xr
import numpy as np
from datetime import datetime

from src.preprocess import CAMELSGBPreprocessor
from src.utils import get_kenya, get_ethiopia

from ..utils import _make_dataset


class TestCAMELSGBPreprocessor:
    def test(tmp_path):
        processor = CAMELSGBPreprocessor(tmp_path)
        assert False
