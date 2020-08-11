import pytest

from tests.utils import _create_dummy_precip_data
from src.analysis import MovingAverage


class TestMovingAverage:
    def test_initialisation(self, tmp_path):
        data_path = _create_dummy_precip_data(tmp_path)
        ma = MovingAverage(data_path / "data_kenya.nc")
        assert ma.name == "3month_moving_average", (
            f"Expected name" f"to be `3month_moving_average` got: {ma.name}"
        )

        with pytest.raises(AttributeError):
            # assert error raised because haven't fit
            ma.index

    def test_fit(self, tmp_path):
        data_path = _create_dummy_precip_data(
            tmp_path, start_date="2000-01-01", end_date="2010-01-01"
        )
        ma = MovingAverage(data_path / "data_kenya.nc")
        variable = "precip"
        ma.fit(variable=variable)

        vars_ = [v for v in ma.index.data_vars]
        assert "precip_3month_moving_average" in vars_, (
            "Expecting `precip_3month_moving_average` variable in" "`self.index`"
        )
