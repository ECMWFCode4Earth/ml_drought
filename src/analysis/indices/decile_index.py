import xarray as xr
import numpy as np

from .base import BaseIndices
from .utils import rolling_cumsum, apply_over_period


class DecileIndex(BaseIndices):
    """https://bit.ly/2NGxIN1

    Calculation:
    -----------
    Monthly precipitation totals from a long-term record are first ranked from
    highest to lowest to construct a cumulative frequency distribution. The
    distribution is then split into 10 parts (tenths of distribution or
    deciles). The first decile is the precipitation value not exceeded by the
    lowest 10% of all precipitation values in a record. The second decile is
    between the lowest 10 and 20% etc. Comparing the amount of precipitation in
    a month (or during a period of several months) with the long-term
    cumulative distribution of precipitation amounts in that period, the
    severity of drought can be assessed.

    Used in Australian drought policy:
        growers and producers are advised to only seek exceptional drought
        assistance if the drought is shown to be an event that occurs only once
        in 20-25 years (deciles 1 and 2 over a 100-year record) and has lasted
        longer than 12 months (White and O’Meagher 1995, Hayes 2000).

    ## References

    Gibbs, W.J. and J.V. Maher. 1967. Rainfall deciles as drought indicators.
     Bureau of Meteorology, Bulletin No. 48, Melbourne.

    Hayes, M.J. 2000. Drought indices. National Drought Mitigation Center,
     University of Nebraska, Lincoln, Nebraska.

    Smith, D.I., M.F. Hutchinson, and R.J. McArthur. 1993. Australian climatic
     and agricultural drought: Payments and policy. Drought Network News
     5(3):11-12.

    White, D.H. and B. O'Meagher. 1995. Coping with exceptional circumstances:
     Droughts in Australia. Drought Network News 7:13–17.
    """

    name = "decile_index"

    @staticmethod
    def bin_to_quintiles(
        da: xr.DataArray, new_variable_name: str = "quintile"
    ) -> xr.Dataset:
        """use the numpy `np.digitize` function to bin the
        variables to quintile labels
        https://stackoverflow.com/a/56514582/9940782

        Arguments:
        ---------
        da : xr.DataArray
            variable that you want to bin into quintiles

        new_variable_name: str
            the `variable_name` in the output `xr.Dataset`
             which corresponds to the quintile labels

        Returns:
        ------
        xr.Dataset

        Note:
            labels = [1, 2, 3, 4, 5] correspond to
             [(0, 20) (20,40) (40,60) (60,80) (80,100)]
        """
        # calculate the quintiles using `np.digitize`
        bins = [0.0, 20.0, 40.0, 60.0, 80.0]
        result = xr.apply_ufunc(np.digitize, da, bins)
        result = result.rename(new_variable_name)
        return result

    @staticmethod
    def rank_norm(ds, dim="time"):
        return (ds.rank(dim=dim) - 1) / (ds.sizes[dim] - 1) * 100

    def fit(
        self, variable: str, time_period: str = "month", rolling_window: int = 3
    ) -> None:
        print("Fitting Decile Index")
        # 1. calculate a cumsum over `rolling_window` timesteps
        ds_window = rolling_cumsum(self.ds, rolling_window)

        # 2. calculate the normalised rank (of each month) for the variable
        out_variable = "rank_norm"
        normalised_rank = apply_over_period(
            ds_window,
            self.rank_norm,
            variable,
            out_variable=out_variable,
            time_str=time_period,
        )
        ds_window = ds_window.merge(normalised_rank.drop("month"))

        # bin the normalised_rank into quintiles
        new_variable_name = "DecileIndex"
        quintile = self.bin_to_quintiles(ds_window[variable], new_variable_name)
        ds_window = ds_window.merge(quintile.to_dataset(name="quintile")).rename(
            {variable: f"{variable}_cumsum"}
        )

        self.index = ds_window
        print("Fitted DI and stored at `obj.index`")
