import xarray as xr
import numpy as np

from src.utils import Region
from src.preprocess.utils import select_bounding_box


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
