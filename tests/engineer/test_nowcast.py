import pytest
# import pickle
import numpy as np
# import xarray as xr

from src.engineer import NowcastEngineer

from ..utils import _make_dataset
from .test_engineer import TestEngineer


class TestNowcastEngineer(TestEngineer):
    def test_init(self, tmp_path):

        with pytest.raises(AssertionError) as e:
            NowcastEngineer(tmp_path)
            assert 'does not exist. Has the preprocesser been run?' in str(e)

        (tmp_path / 'interim').mkdir()

        NowcastEngineer(tmp_path)

        assert (tmp_path / 'features').exists(), 'Features directory not made!'
        assert (tmp_path / 'features' / 'nowcast').exists(), '\
        nowcast directory not made!'

    def test_yearsplit(self, tmp_path):

        self._setup(tmp_path)

        dataset, _, _ = _make_dataset(size=(2, 2))

        engineer = NowcastEngineer(tmp_path)
        train = engineer._train_test_split(dataset, years=[2001],
                                           target_variable='VHI')

        assert (train.time.values < np.datetime64('2001-01-01')).all(), \
            'Got years greater than the test year in the training set!'
