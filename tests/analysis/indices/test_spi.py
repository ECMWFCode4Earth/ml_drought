import pytest
import numpy as np

from tests.utils import _create_dummy_precip_data
from src.analysis.indices import SPI


class TestSPI:
    def test_initialisation(self, tmp_path):
        data_path = _create_dummy_precip_data(tmp_path)
        spi = SPI(data_path / "data_kenya.nc")
        assert spi.name == "spi", f"Expected name" f"to be `spi` got: {spi.name}"

        with pytest.raises(AttributeError):
            # assert error raised because haven't fit
            spi.index

    def test_fit(self, tmp_path):
        data_path = _create_dummy_precip_data(
            tmp_path, start_date="2000-01-01", end_date="2010-01-01"
        )
        assert data_path.exists(), f"file not created correctly"
        spi = SPI(data_path / "data_kenya.nc")
        variable = "precip"
        spi.fit(variable=variable)

        coords = [c for c in spi.index.coords]
        vars_ = [v for v in spi.index.variables if v not in coords]
        assert "SPI3" in vars_, f"Expecting `v` variable in" "`self.index`"

        assert spi.index.SPI3.isel(time=slice(0, 2)).isnull().all(), (
            "" "The first two timesteps should be nan because using SPI3"
        )

        assert not spi.index.SPI3.isel(time=slice(0, 3)).isnull().all(), (
            "" "The first two timesteps but not 3rd should be nan SPI3"
        )

        assert np.isclose(spi.index.SPI3.mean(), 0, atol=0.01), (
            ""
            "Expect the mean SPI value to be close to 0 because"
            "converted to a standard normal distribution"
        )
        assert np.isclose(spi.index.SPI3.std(), 1, atol=0.01), (
            ""
            "Expect the `std()` SPI value to be close to 1 because"
            "converted to a standard normal distribution"
        )

    def test_different_scales(self, tmp_path):
        data_path = _create_dummy_precip_data(
            tmp_path, start_date="2000-01-01", end_date="2010-01-01"
        )
        spi = SPI(data_path / "data_kenya.nc")
        variable = "precip"

        # SPI 6
        spi.fit(variable=variable, scale=6)

        assert spi.index.SPI6.isel(time=slice(0, 5)).isnull().all(), (
            "" "The first two timesteps should be nan because using SPI6"
        )

        assert np.isclose(spi.index.SPI6.mean(), 0, atol=0.01), (
            ""
            "Expect the mean SPI6 value to be close to 0 because"
            "converted to a standard normal distribution"
        )
        assert np.isclose(spi.index.SPI6.std(), 1, atol=0.01), (
            ""
            "Expect the `std()` SPI6 value to be close to 0 because"
            "converted to a standard normal distribution"
        )
