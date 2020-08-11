import xarray as xr

from src.analysis.indices.base import BaseIndices
from tests.utils import _create_dummy_precip_data


class TestBase:
    def test_initialisation(self, tmp_path):
        data_path = _create_dummy_precip_data(tmp_path)
        b = BaseIndices(data_path / "data_kenya.nc")
        assert isinstance(b.ds, xr.Dataset), (
            f"expected to have loaded" f"`ds` as an attribute in `BaseIndices` object"
        )

    def test_save(self, tmp_path):
        file_path = _create_dummy_precip_data(tmp_path)
        data_path = tmp_path / "data"

        if not data_path.exists():
            data_path.mkdir(parents=True, exist_ok=True)

        b = BaseIndices(file_path / "data_kenya.nc")
        b.name = "SPI"
        b.index = xr.open_dataset(file_path / "data_kenya.nc")
        b.save(data_path)

        assert (
            data_path / "analysis" / "indices" / "SPI.nc"
        ).exists(), "Expected to have created a new `.nc` file"
