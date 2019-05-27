import xarray as xr
import numpy as np
import pytest

from src.utils import Region
from src.preprocess.utils import select_bounding_box, regrid


def _make_dataset(size):
    # build dummy .nc object
    height, width = size
    lonmin, lonmax = -180.0, 180.0
    latmin, latmax = -55.152, 75.024

    # extract the size of the lat/lon coords
    lat_len, lon_len = height, width
    # create the vector
    longitudes = np.linspace(lonmin, lonmax, lon_len)
    latitudes = np.linspace(latmin, latmax, lat_len)

    vhi = np.random.randint(100, size=size)

    ds = xr.Dataset(
        {'VHI': (['lat', 'lon'], vhi)},
        coords={
            'lat': latitudes,
            'lon': longitudes}
    )

    return ds, (lonmin, lonmax), (latmin, latmax)


class TestSelectBoundingBox:

    def test_select_bounding_box_inversed(self):
        """Test that inversion works correctly
        """
        size = (100, 100)
        ds, (lonmin, lonmax), (latmin, latmax) = _make_dataset(size)

        global_region = Region(name='global', lonmin=lonmax, lonmax=lonmin,
                               latmin=latmin, latmax=latmax)
        subset = select_bounding_box(ds, global_region, inverse_lon=True)
        assert subset.VHI.values.shape == size, \
            f'Expected output subset to have size {size}, got {subset.VHI.values.shape}'

    def test_selection(self):

        size = (100, 100)
        ds, (lonmin, lonmax), (latmin, latmax) = _make_dataset(size)

        mid = lonmin + ((lonmax - lonmin) / 2)
        half_region = Region(name='half', lonmin=lonmin, lonmax=mid,
                             latmin=latmin, latmax=latmax)
        subset = select_bounding_box(ds, half_region)

        assert subset.VHI.values.shape == (100, 50), \
            f'Expected output subset to have size (50, 100), got {subset.VHI.values.shape}'
        assert max(subset.lon.values) < 0, \
            f'Got a longitude greater than 0, {max(subset.lon.values)}'


class TestRegridding:

    def test_regridding(self):

        size_reference = (100, 100)
        size_target = (1000, 1000)

        reference_ds, _, _ = _make_dataset(size_reference)
        target_ds, _, _ = _make_dataset(size_target)

        regridded_ds = regrid(target_ds, reference_ds)

        assert regridded_ds.VHI.values.shape == size_reference, \
            f'Expected regridded Dataset to have shape {size_reference}, ' \
            f'got {regridded_ds.VHI.values.shape}'

    def test_incorrect_method(self):
        size_reference = (100, 100)
        size_target = (1000, 1000)

        reference_ds, _, _ = _make_dataset(size_reference)
        target_ds, _, _ = _make_dataset(size_target)

        with pytest.raises(AssertionError) as e:
            regrid(target_ds, reference_ds, method='woops!')
        expected_message_contains = 'not an acceptable regridding method. Must be one of'
        assert expected_message_contains in str(e), \
            f'Expected {e} to contain {expected_message_contains}'
