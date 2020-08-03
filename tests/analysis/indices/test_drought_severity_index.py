import pytest

from tests.utils import _create_dummy_precip_data
from src.analysis.indices import DroughtSeverityIndex


class TestDroughtSeverityIndex:
    def test_initialisation(self, tmp_path):
        data_path = _create_dummy_precip_data(tmp_path)
        dsi = DroughtSeverityIndex(data_path / "data_kenya.nc")
        assert dsi.name == "drought_severity_index", (
            f"Expected name" f"to be `drought_severity_index` got: {dsi.name}"
        )

        with pytest.raises(AttributeError):
            # assert error raised because haven't fit
            dsi.index

    def test_fit(self, tmp_path):
        data_path = _create_dummy_precip_data(
            tmp_path, start_date="2000-01-01", end_date="2010-01-01"
        )
        dsi = DroughtSeverityIndex(data_path / "data_kenya.nc")
        variable = "precip"
        dsi.fit(variable=variable)

        coords = [c for c in dsi.index.coords]
        vars_ = [v for v in dsi.index.variables if v not in coords]
        assert "DSI" in vars_, f"Expecting `v` variable in" "`self.index`"

        assert (dsi.index.DSI.max() <= 4) and (dsi.index.DSI.min() >= -4), (
            "" "Range of valid DSI values from -4 to 4"
        )
