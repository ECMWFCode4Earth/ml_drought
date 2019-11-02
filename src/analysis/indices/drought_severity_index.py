import xarray as xr

from .base import BaseIndices
from .utils import rolling_cumsum, apply_over_period


class DroughtSeverityIndex(BaseIndices):
    """
    Calculation:
    -----------
    the raw monthly rainfall totals are integrated to rolling n-monthly totals
    which are then ranked into percentiles by month and this is rescaled to
    range between -4 and +4 in keeping with the range of the Palmer Index.

    The default threshold at -1 which is at 3/8ths or 37.5th percentile.

    ## References

    Smith, D. I, Hutchinson, M. F, & McArthur, R. J. (1992) Climatic and
    Agricultural Drought: Payments and Policy. (Centre for Resource and
    Environmental Studies, Australian National University, Canberra, Australia).

    Hanigan, IC. 2012. The Hutchinson Drought Index Algorithm
    [Computer Software].
       https://github.com/ivanhanigan/HutchinsonDroughtIndex

    Sivakumar, M. V. K., Stone, R., Sentelhas, P. C., Svoboda, M.,Omondi, P.,
    Sarkar, J., and Wardlow, B.: Agricultural Drought Indices: summary and
    recommendations, in: Agricultural Drought Indices, Proceedings of the WMO
    /UNISDR Expert Group Meeting on Agricultural Drought Indices,
    edited by: Sivakumar, M. V.K., Motha, R. P., Wilhite, D. A., and Wood, D. A.,
    2–4 June 2010,AGM-11, WMO/TD No. 1572, WAOB-2011, World Meteorological Organisation,
    Murcia, Spain, Geneva, Switzerland,182 pp.,2010.
    """

    name = "drought_severity_index"

    @staticmethod
    def DSI(da: xr.DataArray, dim: str = "time") -> xr.DataArray:
        y = (da.rank(dim=dim) - 1.0) / (da.sizes[dim] - 1.0)
        z = 8.0 * (y - 0.5)
        return z

    def fit(
        self, variable: str, time_period: str = "month", rolling_window: int = 3
    ) -> None:

        print("Fitting Hutchinson Drought Severity Index")
        ds_window = rolling_cumsum(self.ds, rolling_window)

        out_variable = "DSI"
        dsi = apply_over_period(
            ds_window,
            func=self.DSI,
            in_variable=variable,
            out_variable=out_variable,
            time_str=time_period,
        )
        ds_window = ds_window.merge(dsi.drop("month")).rename(
            {variable: f"{variable}_cumsum"}
        )

        self.index = ds_window
        print("Fitted Drought Severity Index and stored at `obj.index`")
