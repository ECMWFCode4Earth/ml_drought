import pytest

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
