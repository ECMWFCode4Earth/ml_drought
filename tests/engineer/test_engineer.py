import pytest
import pickle
import numpy as np
import xarray as xr

from src.engineer import Engineer

from ..utils import _make_dataset


class TestEngineer:

    def _setup(self, data_path):
        # setup
        interim_folder = data_path / 'interim'
        interim_folder.mkdir()

        expected_output, expected_vars = [], []
        for var in ['a', 'b']:
            (interim_folder / f'{var}_preprocessed').mkdir()

            # this file should be captured
            data, _, _ = _make_dataset((10, 10), var, const=True)
            filename = interim_folder / f'{var}_preprocessed/hello.nc'
            data.to_netcdf(filename)

            expected_output.append(filename)
            expected_vars.append(var)

            # this file should not
            (interim_folder / f'{var}_preprocessed/boo').touch()

        # none of this should be captured
        (interim_folder / 'woops').mkdir()
        woops_data, _, _ = _make_dataset((10, 10), 'oops')
        woops_data.to_netcdf(interim_folder / 'woops/hi.nc')

        return expected_output, expected_vars

    def test_init(self, tmp_path):

        with pytest.raises(AssertionError) as e:
            Engineer(tmp_path)
            assert 'does not exist. Has the preprocesser been run?' in str(e)

        (tmp_path / 'interim').mkdir()

        Engineer(tmp_path)

        assert (tmp_path / 'features').exists(), 'Features directory not made!'

    def test_get_preprocessed(self, tmp_path):

        expected_files, expected_vars = self._setup(tmp_path)

        engineer = Engineer(tmp_path)
        files = engineer._get_preprocessed_files()

        assert set(expected_files) == set(files), f'Did not retrieve expected files!'

    def test_join(self, tmp_path):

        expected_files, expected_vars = self._setup(tmp_path)

        engineer = Engineer(tmp_path)
        joined_ds = engineer._make_dataset()

        dims = ['lon', 'lat', 'time']
        output_vars = [var for var in joined_ds.variables if var not in dims]

        assert set(output_vars) == set(expected_vars), \
            f'Did not retrieve all the expected variables!'

    def test_yearsplit(self, tmp_path):

        self._setup(tmp_path)

        dataset, _, _ = _make_dataset(size=(2, 2))

        engineer = Engineer(tmp_path)
        train = engineer._train_test_split(dataset, years=[2001],
                                           target_variable='VHI')

        assert (train.time.values < np.datetime64('2001-01-01')).all(), \
            'Got years greater than the test year in the training set!'

    def test_engineer(self, tmp_path):

        self._setup(tmp_path)

        engineer = Engineer(tmp_path)
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
            assert np.count_nonzero(norm_dict[key]['mean']) == 0, \
                f'Mean incorrectly calculated!'
            assert np.count_nonzero(norm_dict[key]['std']) == 0, \
                f'Std incorrectly calculated!'
