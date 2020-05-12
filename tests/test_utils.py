import xarray as xr
from src.utils import get_modal_value_across_time
from .utils import _make_dataset


class TestUtils:
    def test_get_modal_value_across_time(self):
        da = _make_dataset((5, 5), const=True)[0].VHI
        da2 = xr.ones_like(da) * 5  # constant of 5s
        modal_da = get_modal_value_across_time(da)

        assert len(modal_da.shape) == 2, (
            "Expecting to have marginalised time"
            "leaving only spatial dimensions (lat, lon)"
        )

        merge_da = xr.concat([da.isel(time=slice(0, 1)), da2], dim="time")
        modal_da2 = get_modal_value_across_time(merge_da)

        assert (modal_da2 == 5).all().values, (
            "Expect that one timestep of 1s"
            "and 36 timesteps of 5s would create a modal_da of 5 for each pixel"
        )
