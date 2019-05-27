import pytest

from .test_utils import _make_dataset

from src.preprocess.base import BasePreProcessor


class TestRegridding:

    def test_regridding(self, tmp_path):

        size_reference = (100, 100)
        size_target = (1000, 1000)

        reference_ds, _, _ = _make_dataset(size_reference)
        target_ds, _, _ = _make_dataset(size_target)

        processor = BasePreProcessor(tmp_path)
        regridded_ds = processor.regrid(target_ds, reference_ds)

        assert regridded_ds.VHI.values.shape == size_reference, \
            f'Expected regridded Dataset to have shape {size_reference}, ' \
            f'got {regridded_ds.VHI.values.shape}'

    def test_incorrect_method(self, tmp_path):
        size_reference = (100, 100)
        size_target = (1000, 1000)

        reference_ds, _, _ = _make_dataset(size_reference)
        target_ds, _, _ = _make_dataset(size_target)

        processor = BasePreProcessor(tmp_path)
        with pytest.raises(AssertionError) as e:
            processor.regrid(target_ds, reference_ds, method='woops!')
        expected_message_contains = 'not an acceptable regridding method. Must be one of'
        assert expected_message_contains in str(e), \
            f'Expected {e} to contain {expected_message_contains}'

    def test_regridder_save(self, tmp_path):
        size_reference = (100, 100)
        size_target = (1000, 1000)

        reference_ds, _, _ = _make_dataset(size_reference)
        target_ds, _, _ = _make_dataset(size_target)

        processor = BasePreProcessor(tmp_path)
        processor.regrid(target_ds, reference_ds)
        assert (processor.interim_folder / 'nearest_s2d_1000x1000_100x100.nc').exists() is False, \
            f'Regridder weight file not deleted!'
