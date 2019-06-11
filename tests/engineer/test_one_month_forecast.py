import pytest
import pickle
import numpy as np
import xarray as xr

from src.engineer import OneMonthForecast

from ..utils import _make_dataset
from .test_engineer import TestEngineer


class TestOneMonthForecast(TestEngineer):
    def test_init(self, tmp_path):

        with pytest.raises(AssertionError) as e:
            OneMonthForecast(tmp_path)
            assert 'does not exist. Has the preprocesser been run?' in str(e)

        (tmp_path / 'interim').mkdir()

        OneMonthForecast(tmp_path)

        assert (tmp_path / 'features').exists(), 'Features directory not made!'
        assert (tmp_path / 'features' / 'one_month_forecast').exists(), '\
        one_month_forecast directory not made!'

    def test_yearsplit(self, tmp_path):

        self._setup(tmp_path)

        dataset, _, _ = _make_dataset(size=(2, 2))

        engineer = OneMonthForecast(tmp_path)
        train = engineer._train_test_split(dataset, years=[2001],
                                           target_variable='VHI')

        assert (train.time.values < np.datetime64('2001-01-01')).all(), \
            'Got years greater than the test year in the training set!'

    def test_engineer(self, tmp_path):

        self._setup(tmp_path)

        engineer = OneMonthForecast(tmp_path)
        engineer.engineer(test_year=2001, target_variable='a')

        def check_folder(folder_path):
            y = xr.open_dataset(folder_path / 'y.nc')
            assert 'b' not in set(y.variables), 'Got unexpected variables in test set'

            x = xr.open_dataset(folder_path / 'x.nc')
            for expected_var in {'a', 'b'}:
                assert expected_var in set(x.variables), \
                    'Missing variables in testing input dataset'
            assert len(x.time.values) == 11, 'Wrong number of months in the test x dataset'
            assert len(y.time.values) == 1, 'Wrong number of months in test y dataset'

        check_folder(tmp_path / 'features/train/1999_12')
        for month in range(1, 13):
            check_folder(tmp_path / f'features/test/2001_{month}')
            check_folder(tmp_path / f'features/train/2000_{month}')

        assert len(list((tmp_path / 'features/train').glob('2001_*'))) == 0, \
            'Test data in the training data!'

        assert (tmp_path / 'features/normalizing_dict.pkl').exists(), \
            f'Normalizing dict not saved!'
        with (tmp_path / 'features/normalizing_dict.pkl').open('rb') as f:
            norm_dict = pickle.load(f)

        for key, val in norm_dict.items():
            assert key in {'a', 'b'}, f'Unexpected key!'
            assert (norm_dict[key]['mean'] == 1).all(), \
                f'Mean incorrectly calculated!'
            assert (norm_dict[key]['std'] == 0).all(), \
                f'Std incorrectly calculated!'
