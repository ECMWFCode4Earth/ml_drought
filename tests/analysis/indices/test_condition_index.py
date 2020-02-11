import numpy as np
import pytest

from tests.utils import _create_dummy_precip_data
from src.analysis.indices import ConditionIndex


class TestConditionIndex:
    def test_initialisation(self, tmp_path):
        data_path = _create_dummy_precip_data(tmp_path)
        ci = ConditionIndex(data_path / "chirps_kenya.nc")
        assert ci.name == "decile_index", (
            f"Expected name" f"to be `decile_index` got: {ci.name}"
        )

        with pytest.raises(AttributeError):
            # assert error raised because haven't fit
            ci.index

    def test_fit(self, tmp_path):
        data_path = _create_dummy_precip_data(
            tmp_path, start_date="2000-01-01", end_date="2010-01-01"
        )
        ci = ConditionIndex(data_path / "chirps_kenya.nc")
        variable = "precip"
        ci.fit(variable=variable)

        coords = [c for c in ci.index.coords]
        vars_ = [v for v in ci.index.variables if v not in coords]
        assert "quintile" in vars_, f"Expecting `quintile` variable in" "`self.index`"

        vals = np.unique(ci.index.quintile.values)
        assert all(np.isin(np.arange(1, 6), vals)), (
            f"Expect quintiles to" f" be from 1 - 5. Got: {vals}"
        )

        assert ci.index.rank_norm.min().values == 0, (
            f"Expect minmum "
            f"rank_norm to be 0. Got: {ci.index.rank_norm.min().values}"
        )

        assert ci.index.rank_norm.max().values == 100, (
            f"Expect max "
            f"rank_norm to be 100. Got: {ci.index.rank_norm.max().values}"
        )
