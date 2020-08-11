import numpy as np
import pytest

from tests.utils import _create_dummy_precip_data
from src.analysis.indices import VegetationDeficitIndex


class TestVegetationDefictonditionIndex:
    def test_initialisation(self, tmp_path):
        data_path = _create_dummy_precip_data(tmp_path)
        vdi = VegetationDeficitIndex(data_path / "data_kenya.nc")
        assert vdi.name == "vegetation_deficit_index", (
            "Expected name" f"to be `vegetation_deficit_index` got: {vdi.name}"
        )

        with pytest.raises(AttributeError):
            # assert error raised because haven't fit
            vdi.index

    def test_fit(self, tmp_path):
        data_path = _create_dummy_precip_data(
            tmp_path, start_date="2000-01-01", end_date="2010-01-01"
        )
        vdi = VegetationDeficitIndex(data_path / "data_kenya.nc")
        variable = "precip"
        vdi.fit(variable=variable)

        vars_ = [v for v in vdi.index.data_vars]
        assert "VCI3M_index" in vars_, (
            f"Expecting `VCI3M_index` variable in" "`self.index`"
        )

        vals = np.unique(vdi.index.VCI3M_index.values)
        assert all(np.isin(np.arange(1, 6), vals)), (
            f"Expect quintiles to" f" be from 1 - 5. Got: {vals}"
        )
