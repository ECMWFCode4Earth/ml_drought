import pytest

from src.analysis.indices import AnomalyIndex
from tests.utils import _create_dummy_precip_data


class TestAnomalyIndex:
    def test_initialisation(self, tmp_path):
        data_path = _create_dummy_precip_data(tmp_path)
        a = AnomalyIndex(data_path / "data_kenya.nc")
        assert a.name == "rainfall_anomaly_index", (
            f"Expected name" f"to be `rainfall_anomaly_index` got: {a.name}"
        )

        with pytest.raises(AttributeError):
            # assert error raised because haven't fit
            a.index

    def test_fit(self, tmp_path):
        data_path = _create_dummy_precip_data(
            tmp_path, start_date="2000-01-01", end_date="2010-01-01"
        )
        ai = AnomalyIndex(data_path / "data_kenya.nc")
        variable = "precip"
        ai.fit(variable=variable)

        coords = [c for c in ai.index.coords]
        vars_ = [v for v in ai.index.variables if v not in coords]
        assert "RAI" in vars_, f"Expecting `RAI` variable in `self.index`"
