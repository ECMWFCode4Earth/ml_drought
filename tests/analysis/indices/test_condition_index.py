import numpy as np
import pytest

from tests.utils import _create_dummy_precip_data
from src.analysis.indices import ConditionIndex


class TestConditionIndex:
    def test_initialisation(self, tmp_path):
        data_path = _create_dummy_precip_data(tmp_path)
        ci = ConditionIndex(data_path / "data_kenya.nc")
        assert ci.name == "condition_index", (
            f"Expected name" f"to be `condition_index` got: {ci.name}"
        )

        with pytest.raises(AttributeError):
            # assert error raised because haven't fit
            ci.index

    def test_fit(self, tmp_path):
        data_path = _create_dummy_precip_data(
            tmp_path, start_date="2000-01-01", end_date="2010-01-01"
        )
        ci = ConditionIndex(data_path / "data_kenya.nc")
        variable = "precip"
        ci.fit(variable=variable)

        coords = [c for c in ci.index.coords]
        vars_ = [v for v in ci.index.variables if v not in coords]
        assert "precip_condition_index_1" in vars_, (
            f"Expecting `precip_condition_index_1` variable in" "`self.index`"
        )

        # MIN-MAX of 0, 100
        assert np.isclose(
            float(ci.index[vars_[0]].max().values), 100
        ), f"Expected to have max value of 100. Got: {ci.index[vars_[0]].max()}"
        assert np.isclose(
            float(ci.index[vars_[0]].min().values), 0
        ), f"Expected to have min value of 0. Got: {ci.index[vars_[0]].min()}"

        assert (
            np.isclose(ci.index[vars_[0]].max(dim="time"), 100)
        ).all(), "Expect all times to have a max value of 100"
        assert (
            np.isclose(ci.index[vars_[0]].max(dim=["lat", "lon"]), 100)
        ).all(), "Expect all pixels (lat, lon) to have a max value of 100"
        assert (
            np.isclose(ci.index[vars_[0]].min(dim="time"), 0)
        ).all(), "Expect all times to have a min value of 0"
        assert (
            np.isclose(ci.index[vars_[0]].min(dim=["lat", "lon"]), 0)
        ).all(), "Expect all pixels (lat, lon) to have a min value of 0"
