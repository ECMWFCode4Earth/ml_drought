import pytest
import numpy as np

from tests.utils import _create_dummy_precip_data
from src.analysis.indices import PercentNormalIndex


class TestPercentNormalIndex:
    def test_initialisation(self, tmp_path):
        data_path = _create_dummy_precip_data(tmp_path)
        pni = PercentNormalIndex(data_path / "data_kenya.nc")
        assert pni.name == "percent_normal_index", (
            f"Expected name" f"to be `percent_normal_index` got: {pni.name}"
        )

        with pytest.raises(AttributeError):
            # assert error raised because haven't fit
            pni.index

    def test_fit(self, tmp_path):
        data_path = _create_dummy_precip_data(
            tmp_path, start_date="2000-01-01", end_date="2010-01-01"
        )
        pni = PercentNormalIndex(data_path / "data_kenya.nc")
        variable = "precip"
        pni.fit(variable=variable)

        coords = [c for c in pni.index.coords]
        vars_ = [v for v in pni.index.variables if v not in coords]
        assert "PNI" in vars_, f"Expecting `v` variable in" "`self.index`"

        assert np.isclose(pni.index.PNI.mean().values, 100, atol=5), (
            "" "Expect the mean PNI value to be 100 (100% is NORMAL)"
        )
