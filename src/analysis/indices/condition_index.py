import xarray as xr

from .base import BaseIndices
from .utils import rolling_mean


class ConditionIndex(BaseIndices):
    """
    Condition Index
        (Vegetation Condition Index, Rainfall Condition Index)

    Calculation
    -----------
    100 * (value_{pixel, time} - value_{pixel_min}) / (value_{pixel_max} - value_{pixel_min})

    WARNING: Relative Range indices - they are susceptible to occurences of
    extreme values in the data, because they are scaled by the MAX / MIN
    observed values.

    ## References

    Kogan, F.N., 1997: Global Drought Watch from Space.
    Bulletin of the American Meteorological Society, 78, 621-636.

    Adede, Chrisgone, et al. "A Mixed Model Approach to
    Vegetation Condition Prediction Using Artificial
    Neural Networks (ANN): Case of Kenya’s Operational
    Drought Monitoring." Remote Sensing 11.9 (2019): 1099.

    """

    name = "condition_index"

    @staticmethod
    def condition_index(da: xr.DataArray, dim: str = "time") -> xr.DataArray:
        # calculate the Monthly min/max (normalise over months AND pixels)
        monmin = da.groupby("time.month").min(dim="time")
        monmax = da.groupby("time.month").max(dim="time")
        # copy array forwards through time
        sameshape_monmin = monmin.sel(month=da["time.month"]).drop("month")
        sameshape_monmax = monmax.sel(month=da["time.month"]).drop("month")

        # The max/min for each unique pixel-month
        relative_range = (da - sameshape_monmin) / (sameshape_monmax - sameshape_monmin)
        condition_index = 100 * relative_range

        return condition_index

    def fit(self, variable: str, rolling_window: int = 1) -> None:

        var_name = f"{variable}_{self.name}_{rolling_window}"
        print(f"Fitting {variable} Condition Index")
        assert rolling_window > 0, "Must have a rolling window > 0"

        condition_index = self.condition_index(self.ds[variable])
        # NOTE: if rolling_window = 1 the calculation doesn't change values
        rolling_ds = rolling_mean(condition_index, rolling_window)

        self.index = rolling_ds.to_dataset(name=var_name)
        print(f"Fitted {variable} Condition Index and stored at `obj.index`")


"""
TESTS:

# check the mean for EACH PIXEL is close to 50
assert np.isclose(condition_index.isel(lat=0, lon=0).mean(), 50)

# check the max for EACH PIXEL is 100
assert condition_index.isel(lat=0, lon=0).max() == 100

# check the min for EACH PIXEL is 0
assert condition_index.isel(lat=0, lon=0).min() == 0

# check there are multiple 100 values for each pixel (for each month)
assert (
    condition_index
    .isel(lat=0, lon=0)  # first pixel
    .sel(time=condition_index.isel(lat=0, lon=0) == 100)  # where value is 100
    .values.shape
) == condition_index.isel(lat=0, lon=0)['time.month'].sel(time=condition_index.isel(lat=0, lon=0)['time.month'] == 11).values.shape
"""
