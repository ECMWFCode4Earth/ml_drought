from src.utils import Region
from src.preprocess.utils import select_bounding_box
from ..utils import _make_dataset


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
        assert subset.VHI.values.shape[1:] == size, \
            f'Expected output subset to have size {size}, got {subset.VHI.values.shape[1:]}'

    def test_selection(self):

        size = (100, 100)
        ds, (lonmin, lonmax), (latmin, latmax) = _make_dataset(size)

        mid = lonmin + ((lonmax - lonmin) / 2)
        half_region = Region(name='half', lonmin=lonmin, lonmax=mid,
                             latmin=latmin, latmax=latmax)
        subset = select_bounding_box(ds, half_region)

        assert subset.VHI.values.shape[1:] == (100, 50), \
            f'Expected output subset to have size (50, 100), got {subset.VHI.values.shape}'
        assert max(subset.lon.values) < 0, \
            f'Got a longitude greater than 0, {max(subset.lon.values)}'


# class TestSHPToNetCDF:
