import pytest

from ..utils import _make_dataset

from src.preprocess.base import BasePreProcessor


class TestBase:

    def test_resampling(self):
        monthly_in, _, _ = _make_dataset(size=(10, 10))

        monthly = BasePreProcessor.resample_time(monthly_in, resample_length='M')

        assert len(monthly_in.time) == len(monthly.time)

    def test_regridding(self, tmp_path):

        size_reference = (10, 10)
        size_target = (20, 20)

        reference_ds, _, _ = _make_dataset(size_reference)
        target_ds, _, _ = _make_dataset(size_target)

        processor = BasePreProcessor(tmp_path)
        regridded_ds = processor.regrid(target_ds, reference_ds)

        # add the time dimension
        assert regridded_ds.VHI.values.shape[1:] == size_reference, \
            f'Expected regridded Dataset to have shape {size_reference}, ' \
            f'got {regridded_ds.VHI.values.shape}'

    def test_incorrect_method(self, tmp_path):
        size_reference = (10, 10)
        size_target = (100, 100)

        reference_ds, _, _ = _make_dataset(size_reference)
        target_ds, _, _ = _make_dataset(size_target)

        processor = BasePreProcessor(tmp_path)
        with pytest.raises(AssertionError) as e:
            processor.regrid(target_ds, reference_ds, method='woops!')
        expected_message_contains = 'not an acceptable regridding method. Must be one of'
        assert expected_message_contains in str(e), \
            f'Expected {e} to contain {expected_message_contains}'

    def test_regridder_save(self, tmp_path):
        size_reference = (10, 10)
        size_target = (20, 20)

        reference_ds, _, _ = _make_dataset(size_reference)
        target_ds, _, _ = _make_dataset(size_target)

        processor = BasePreProcessor(tmp_path)
        processor.regrid(target_ds, reference_ds)
        weight_filename = 'nearest_s2d_100x100_10x10.nc'
        assert (processor.preprocessed_folder / weight_filename).exists() is False, \
            f'Regridder weight file not deleted!'

    def test_load_regridder(self, tmp_path):

        test_dataset, _, _ = _make_dataset(size=(10, 10))
        test_dataset.to_netcdf(tmp_path / 'regridder.nc')

        output = BasePreProcessor.load_reference_grid(tmp_path / 'regridder.nc')

        assert set(output.variables) == {'lat', 'lon'}, f'Got extra variables: {output.variables}'
