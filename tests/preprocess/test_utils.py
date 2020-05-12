import pytest
import numpy as np

from src.utils import Region
from src.preprocess.utils import select_bounding_box, SHPtoXarray
from ..utils import _make_dataset, CreateSHPFile


class TestSelectBoundingBox:
    def test_select_bounding_box_inversed(self):
        """Test that inversion works correctly
        """
        size = (100, 100)
        ds, (lonmin, lonmax), (latmin, latmax) = _make_dataset(size)

        global_region = Region(
            name="global", lonmin=lonmax, lonmax=lonmin, latmin=latmin, latmax=latmax
        )
        subset = select_bounding_box(ds, global_region, inverse_lon=True)

        # add the time dimension
        assert (
            subset.VHI.values.shape[1:] == size
        ), f"Expected output subset to have size {size}, got {subset.VHI.values.shape[1:]}"

    def test_selection(self):

        size = (100, 100)
        ds, (lonmin, lonmax), (latmin, latmax) = _make_dataset(size)

        mid = lonmin + ((lonmax - lonmin) / 2)
        half_region = Region(
            name="half", lonmin=lonmin, lonmax=mid, latmin=latmin, latmax=latmax
        )
        subset = select_bounding_box(ds, half_region)

        assert subset.VHI.values.shape[1:] == (
            100,
            50,
        ), f"Expected output subset to have size (50, 100), got {subset.VHI.values.shape}"
        assert (
            max(subset.lon.values) < 0
        ), f"Got a longitude greater than 0, {max(subset.lon.values)}"


class TestSHPtoXarray:
    @pytest.mark.xfail(reason="geopandas not part of the testing environment")
    def test_shapefile_to_xarray(self, tmp_path):
        shp_filepath = (
            tmp_path
            / "raw"
            / "boundaries"
            / "kenya"
            / "Admin2/KEN_admin2_2002_DEPHA.shp"
        )
        shp_filepath.parents[0].mkdir(parents=True, exist_ok=True)
        s = CreateSHPFile()
        s.create_demo_shapefile(shp_filepath)

        reference_ds, _, _ = _make_dataset(
            size=(30, 30), latmin=-1, latmax=1, lonmin=33, lonmax=35
        )

        ds = SHPtoXarray().shapefile_to_xarray(
            da=reference_ds.VHI,
            shp_path=shp_filepath,
            var_name="testy_test",
            lookup_colname="PROVINCE",
        )

        assert "testy_test" in ds.data_vars
        # check the attrs are correctly assigned
        assert np.isin(
            ["keys", "values", "unique_values"], (list(ds.attrs.keys()))
        ).all()
