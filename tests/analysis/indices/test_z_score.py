import pytest
import numpy as np

from tests.utils import _create_dummy_precip_data
from src.analysis.indices import ZScoreIndex


class TestZScoreIndex:
    def test_initialisation(self, tmp_path):
        data_path = _create_dummy_precip_data(tmp_path)
        zsi = ZScoreIndex(data_path / "data_kenya.nc")
        assert zsi.name == "z_score_index", (
            f"Expected name" f"to be `z_score_index` got: {zsi.name}"
        )

        with pytest.raises(AttributeError):
            # assert error raised because haven't fit
            zsi.index

    def test_fit(self, tmp_path):
        data_path = _create_dummy_precip_data(
            tmp_path, start_date="2000-01-01", end_date="2010-01-01"
        )
        zsi = ZScoreIndex(data_path / "data_kenya.nc")
        variable = "precip"
        zsi.fit(variable=variable)

        coords = [c for c in zsi.index.coords]
        vars_ = [v for v in zsi.index.variables if v not in coords]
        assert "ZSI" in vars_, f"Expecting `ZSI` variable in" "`self.index`"

        assert np.isclose(zsi.index.ZSI.mean(), 0, atol=0.01), (
            "" "Expect the mean ZScoreIndex value to be close to 0"
        )
