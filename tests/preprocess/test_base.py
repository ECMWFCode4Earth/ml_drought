import pytest
import numpy as np
from pandas import to_datetime

from src.utils import region_lookup
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

    def test_chop_roi(self, tmp_path):
        size_original = (80, 80)
        original_ds, _, _ = _make_dataset(size_original)

        original_shape = original_ds.VHI.shape

        processor = BasePreProcessor(tmp_path)
        subset_str = 'east_africa'
        new_ds = processor.chop_roi(ds=original_ds, subset_str=subset_str)
        output_shape = new_ds.VHI.shape

        assert original_shape != output_shape, f"The chop_roi should lead to\
        smaller datasets than the original. Expected output_shape: {output_shape}\
        to be different from original_shape: {original_shape}"

        assert (
            (new_ds.lat.values.min() >= -11) & (new_ds.lat.values.max() <= 23)
        ), f"Expected latitude to be in the range -11 : 23. Currently:\
        {new_ds.lat.values.min()} : {new_ds.lat.values.max()}"

        assert (
            (new_ds.lon.values.min() >= 21) & (new_ds.lon.values.max() <= 51.8)
        ), f"Expected longitude to be in the range 21 : 51.8. Currently:\
        {new_ds.lon.values.min()} : {new_ds.lon.values.max()}"

    def test_fix_invalid_timesteps(self, tmp_path):
        processor = BasePreProcessor(tmp_path)
        variable = 'VHI'
        subset_str = 'indian_ocean'
        ds = self._make_dataset_with_invalid_data()

        out = processor._fix_invalid_timesteps(
            ds=ds, variable=variable, subset_str=subset_str, invert_lat=False
        )

        assert len(out.time) == len(ds.time) - 3, 'There should be 3 timesteps removed: '\
            '[2000-01-31, 2000-08-31, 2001-08-31]'
        assert not np.isin(
            [to_datetime(t) for t in ds.time.values],
            [to_datetime(t) for t in ['2000-01-31', '2000-08-31', '2001-08-31']]
        ).all()

    @staticmethod
    def _make_dataset_with_invalid_data():
        size = (50, 50)
        ds, _, _ = _make_dataset(size, lonmin=33.501, lonmax=42.283,
                                 latmin=-5.202, latmax=6.002)
        ds = ds.astype('float64')

        lc = ds.coords['lon']
        la = ds.coords['lat']

        region = region_lookup['indian_ocean']
        latmin = region.latmin
        latmax = region.latmax
        lonmin = region.lonmin
        lonmax = region.lonmax

        ds.VHI.loc[
            dict(
                lon=lc[(lc > lonmin) & (lc < lonmax)],
                lat=la[(la > latmin) & (la < latmax)]
            )
        ] = np.nan

        errored_time_values = np.ones(size)

        # create 3 errored timesteps
        ds.VHI.loc['2000-01-31'] = errored_time_values
        ds.VHI.loc['2000-08-31'] = errored_time_values
        ds.VHI.loc['2001-08-31'] = errored_time_values

        return ds
