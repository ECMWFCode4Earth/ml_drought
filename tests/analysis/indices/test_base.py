# import numpy as np
# import pandas as pd
import xarray as xr

from src.analysis.indices.base import BaseIndices
from tests.utils import _create_dummy_precip_data


class TestBase:

    def test_initialisation(self, tmp_path):
        data_path = _create_dummy_precip_data(tmp_path)
        b = BaseIndices(data_path / 'chirps_kenya.nc')
        assert isinstance(b.ds, xr.Dataset), f"expected to have loaded"\
            f"`ds` as an attribute in `BaseIndices` object"
