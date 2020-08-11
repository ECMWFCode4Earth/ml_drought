import numpy as np
import pytest

from src.analysis.indices import ChinaZIndex
from tests.utils import _create_dummy_precip_data


class TestChinaZIndex:
    def test_initialisation(self, tmp_path):
        data_path = _create_dummy_precip_data(tmp_path)
        czi = ChinaZIndex(data_path / "data_kenya.nc")
        assert czi.name == "china_z_index", (
            f"Expected name" f"to be `china_z_index` got: {czi.name}"
        )

        with pytest.raises(AttributeError):
            # assert error raised because haven't fit
            czi.index

    def test_fit(self, tmp_path):
        data_path = _create_dummy_precip_data(
            tmp_path, start_date="2000-01-01", end_date="2010-01-01"
        )
        czi = ChinaZIndex(data_path / "data_kenya.nc")
        variable = "precip"
        czi.fit(variable=variable)

        coords = [c for c in czi.index.coords]
        vars_ = [v for v in czi.index.variables if v not in coords]
        assert "CZI" in vars_, f"Expecting `CZI` variable in `self.index`"
        assert np.isclose(czi.index.CZI.median().values, 0, atol=0.1), (
            ""
            "Because values are normally distributed expect CZI median to be"
            "close to 0"
        )

    def test_fit_modified(self, tmp_path):
        data_path = _create_dummy_precip_data(
            tmp_path, start_date="2000-01-01", end_date="2010-01-01"
        )
        czi = ChinaZIndex(data_path / "data_kenya.nc")
        variable = "precip"
        czi.fit(variable=variable, modified=True)

        coords = [c for c in czi.index.coords]
        vars_ = [v for v in czi.index.variables if v not in coords]
        assert "MCZI" in vars_, f"Expecting `CZI` variable in `self.index`"

        assert np.isclose(czi.index.MCZI.median().values, 0, atol=0.1), (
            ""
            "Because values are normally distributed expect CZI median to be"
            "close to 0"
        )
