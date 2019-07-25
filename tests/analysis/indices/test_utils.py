import xarray as xr
import pandas as pd

from src.analysis.indices.utils import (
    rolling_cumsum,
    apply_over_period,
    create_shape_aligned_climatology,
)
from tests.utils import _create_dummy_precip_data


class TestIndicesUtils:
    def test_rolling_cumsum(self, tmp_path):
        data_path = _create_dummy_precip_data(tmp_path)
        ds = xr.open_dataset(data_path / 'chirps_kenya.nc')

        cumsum3 = rolling_cumsum(ds, rolling_window=3)
        cumsum6 = rolling_cumsum(ds, rolling_window=6)

        # assert start in 2nd month (CENTRED WINDOW)
        got = pd.to_datetime(cumsum3.isel(time=0).time.values)
        expected = pd.to_datetime('1999-02-28')
        assert got == expected, "Expected minimum time of Cumsum to be:" \
            f"{expected} because centered window with size=3. Got: {got}"

        # assert start in 4th month
        got = pd.to_datetime(cumsum6.isel(time=0).time.values)
        expected = pd.to_datetime('1999-04-30')
        assert got == expected, "Expected minimum time of Cumsum to be:"\
            f"{expected} because centered window with size=6. Got: {got}"

    def test_apply_over_period(self, tmp_path):
        data_path = _create_dummy_precip_data(tmp_path)
        ds = xr.open_dataset(data_path / 'chirps_kenya.nc')

        got = apply_over_period(
            ds, xr.Dataset.mean, in_variable='precip',
            out_variable='mean', time_str='month'
        )
        expected = (
            ds
            .groupby('time.month')
            .mean(dim='time')
            .rename({'precip': 'mean'})
        )
        assert all(got == expected), f"Expected: {expected}\nGot: {got}"

    def test_create_shape_aligned_climatology_month(self, tmp_path):
        data_path = _create_dummy_precip_data(tmp_path)
        ds = xr.open_dataset(data_path / 'chirps_kenya.nc')

        time_period = 'month'
        ds_window = rolling_cumsum(ds, rolling_window=3)
        mthly_climatology = (
            ds_window
            .groupby(f'time.{time_period}')
            .mean(dim='time')
        )

        assert mthly_climatology.precip.shape[0] == 12, f"There should be 12" \
            f"month when calculating monthly climatology"

        clim = create_shape_aligned_climatology(
            ds_window, mthly_climatology,
            variable='precip', time_period=time_period
        )

        assert ds_window['precip'].shape == clim['precip'].shape, f"Expected" \
            f"the shape of `ds_window` ({ds_window['precip'].shape}) to ==" \
            f"`clim` ({clim['precip'].shape})"

    def test_create_shape_aligned_climatology_season(self, tmp_path):
        data_path = _create_dummy_precip_data(tmp_path)
        ds = xr.open_dataset(data_path / 'chirps_kenya.nc')

        time_period = 'season'
        ds_window = rolling_cumsum(ds, rolling_window=3)
        climatology = (
            ds_window
            .groupby(f'time.{time_period}')
            .mean(dim='time')
        )

        assert climatology.precip.shape[0] == 4, f"There should be 4 seasons" \
            f"in the climatology"

        clim = create_shape_aligned_climatology(
            ds_window, climatology,
            variable='precip', time_period=time_period
        )
        assert ds_window['precip'].shape == clim['precip'].shape, f"Expected" \
            f"the shape of `ds_window` ({ds_window['precip'].shape}) to ==" \
            f"`clim` ({clim['precip'].shape})"
