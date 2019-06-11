import pytest
# import pickle
# import numpy as np
# import xarray as xr

from src.engineer import Nowcast

# from ..utils import _make_dataset
from .test_engineer import TestEngineer


class TestNowcast(TestEngineer):
    def test_init(self, tmp_path):

        with pytest.raises(AssertionError) as e:
            Nowcast(tmp_path)
            assert 'does not exist. Has the preprocesser been run?' in str(e)

        (tmp_path / 'interim').mkdir()

        Nowcast(tmp_path)

        assert (tmp_path / 'features').exists(), 'Features directory not made!'
        assert (tmp_path / 'features' / 'nowcast').exists(), '\
        nowcast directory not made!'

    pass
