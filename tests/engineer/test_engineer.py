import pytest
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
            data, _, _ = _make_dataset((10, 10), var)
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
        train, test = engineer._train_test_split(dataset, years=[2001],
                                                 target_variable='VHI')

        assert (train.time.values < np.datetime64('2001-01-01')).all(), \
            'Got years greater than the test year in the training set!'

        assert len(test[2001]) == 12, f'Expected 12 testing months in the test dataset'

        for dataset in {'x', 'y'}:
            assert (test[2001][12][dataset].time.values > np.datetime64('2000-12-31')).all(), \
                'Got years smaller than the test year in the test set!'

    def test_engineer(self, tmp_path):

        self._setup(tmp_path)

        engineer = Engineer(tmp_path)
        engineer.engineer(test_year=2001, target_variable='a')

        # first, lets make sure the right files were created

        assert (tmp_path / 'features/train.nc').exists(), \
            f'Training file not generated!'

        for month in range(1, 13):
            for ds in {'x', 'y'}:
                assert (tmp_path / f'features/test_2001_{month}/{ds}.nc').exists(), \
                    f'Test folder not generated!'

        train = xr.open_dataset(tmp_path / 'features/train.nc')
        for expected_var in {'a', 'b'}:
            assert expected_var in set(train.variables), \
                'Missing variables in training set'
        for month in range(1, 13):
            test_y = xr.open_dataset(tmp_path / f'features/test_2001_{month}/y.nc')
            assert 'b' not in set(test_y.variables), 'Got unexpected variables in test set'

            test_x = xr.open_dataset(tmp_path / f'features/test_2001_{month}/x.nc')
            for expected_var in {'a', 'b'}:
                assert expected_var in set(test_x.variables), \
                    'Missing variables in testing input dataset'
            assert len(test_x.time.values) == 11, f'Wrong number of months in the test dataset'
