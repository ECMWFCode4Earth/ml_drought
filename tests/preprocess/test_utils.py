import xarray as xr
import numpy as np

from src.utils import Region
from src.preprocess.utils import select_bounding_box


def _make_dataset(size, lonmin=-180.0, lonmax=180.0,
                  latmin=-55.152, latmax=75.024,
                  add_times=True):

    lat_len, lon_len = size
    # create the vector
    longitudes = np.linspace(lonmin, lonmax, lon_len)
    latitudes = np.linspace(latmin, latmax, lat_len)

    dims = ['lat', 'lon']
    coords = {'lat': latitudes,
              'lon': longitudes}

    if add_times:
        size = (2, size[0], size[1])
        dims.insert(0, 'time')
        coords['time'] = [0, 1]
    vhi = np.random.randint(100, size=size)

    ds = xr.Dataset({'VHI': (dims, vhi)}, coords=coords)

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

        # add the time dimension
        expected_size = (2, 100, 100)
        assert subset.VHI.values.shape == expected_size, \
            f'Expected output subset to have size {expected_size}, got {subset.VHI.values.shape}'

    def test_selection(self):

        size = (100, 100)
        ds, (lonmin, lonmax), (latmin, latmax) = _make_dataset(size)

        mid = lonmin + ((lonmax - lonmin) / 2)
        half_region = Region(name='half', lonmin=lonmin, lonmax=mid,
                             latmin=latmin, latmax=latmax)
        subset = select_bounding_box(ds, half_region)

        assert subset.VHI.values.shape == (2, 100, 50), \
            f'Expected output subset to have size (2, 50, 100), got {subset.VHI.values.shape}'
        assert max(subset.lon.values) < 0, \
            f'Got a longitude greater than 0, {max(subset.lon.values)}'
