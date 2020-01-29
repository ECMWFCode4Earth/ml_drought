import xarray as xr
import numpy as np

from .base import BaseIndices
from .utils import rolling_mean, apply_over_period


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

    name = 'condition_index'

    @staticmethod
    def condition_index(da: xr.DataArray, dim: str = "time") -> xr.DataArray:
        relative_range = (da - da.min(dim=dim)) / (da.max(dim=dim) - da.min(dim=dim))
        condition_index = 100 * relative_range

        return condition_index

    def fit(
        self,
        variable: str,
        rolling_window: int = 1
    ) -> None:

        var_name = f"{variable}_{self.name}_{rolling_window}"
        print(f"Fitting {variable} Condition Index")
        assert rolling_window > 0, 'Must have a rolling window > 0'

        rolling_ds = rolling_mean(self.ds[variable], rolling_window)
        condition_index = self.condition_index(
            rolling_ds
        )

        self.index = condition_index.to_dataset(var_name)
        print(f"Fitted {variable} Condition Index and stored at `obj.index`")
