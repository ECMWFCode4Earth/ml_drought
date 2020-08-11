import numpy as np
import pytest

from tests.utils import _create_dummy_precip_data
from src.analysis.indices import DecileIndex


class TestDecileIndex:
    def test_initialisation(self, tmp_path):
        data_path = _create_dummy_precip_data(tmp_path)
        di = DecileIndex(data_path / "data_kenya.nc")
        assert di.name == "decile_index", (
            f"Expected name" f"to be `decile_index` got: {di.name}"
        )

        with pytest.raises(AttributeError):
            # assert error raised because haven't fit
            di.index

    def test_fit(self, tmp_path):
        data_path = _create_dummy_precip_data(
            tmp_path, start_date="2000-01-01", end_date="2010-01-01"
        )
        di = DecileIndex(data_path / "data_kenya.nc")
        variable = "precip"
        di.fit(variable=variable)

        coords = [c for c in di.index.coords]
        vars_ = [v for v in di.index.variables if v not in coords]
        assert "quintile" in vars_, f"Expecting `quintile` variable in" "`self.index`"

        vals = np.unique(di.index.quintile.values)
        assert all(np.isin(np.arange(1, 6), vals)), (
            f"Expect quintiles to" f" be from 1 - 5. Got: {vals}"
        )

        assert di.index.rank_norm.min().values == 0, (
            f"Expect minmum "
            f"rank_norm to be 0. Got: {di.index.rank_norm.min().values}"
        )

        assert di.index.rank_norm.max().values == 100, (
            f"Expect max "
            f"rank_norm to be 100. Got: {di.index.rank_norm.max().values}"
        )
