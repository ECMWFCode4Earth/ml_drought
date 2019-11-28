import xarray as xr
import numpy as np

from .base import BaseIndices
from .utils import rolling_cumsum, apply_over_period


class ChinaZIndex(BaseIndices):
    """
    used by National Climate Centre (NCC) of China to monitor drought

    Calculation
    -----------
    CZI assumes that precipitation data follow the Pearson Type III
    distribution and is related to Wilson–Hilferty cube-root transformation
    (Wilson and Hilferty, 1931) from chi-square variable to the Z-scale
    (Kendall and Stuart, 1977).

    ## References

    Dogan, S., Berktay, A., Singh, V.P., 2012. Comparison of multi-monthly
     rainfall-based drought severity indices, with application to semi-arid Konya
      closed basin, Turkey. J. Hydrol. 470–471, 255–268.

    Kendall, M.G.; Stuart, A. The Advanced Theory of Statistics; Charles Griffin
     & Company-High Wycombe: London, UK, 1997; pp. 400–401.

    Morid, S., Smakhtin, V., Moghaddasi, M., 2006. Comparison of seven
     meteorological indices for drought monitoring in Iran. Int. J. Climatol.
     26, 971–985.

    Wilson, E.B., Hilferty, M.M., 1931. The Distribution of Chi-Square. Proc.
     Natl. Acad. Sci. USA 17, 684–688.

    Wu, H., Hayes, M.J., Weiss, A., Hu, Q.I., 2001. An evaluation of the
     standardized precipitation index, the china-Zindex and the statistical
     Z-Score. Int. J. Climatol.21, 745–758. http://dx.doi.org/10.1002/joc.658.
    """

    name = "china_z_index"

    @staticmethod
    def MCZI(da: xr.DataArray, dim: str = "time") -> xr.DataArray:
        zsi = (da - da.median(dim=dim)) / da.std(dim=dim)
        cs = np.power(zsi, 3) / da.sizes[dim]
        mczi = (
            6.0 / cs * np.power((cs / 2.0 * zsi + 1.0), 1.0 / 3.0) - 6.0 / cs + cs / 6.0
        )
        return mczi

    @staticmethod
    def CZI(da: xr.DataArray, dim: str = "time") -> xr.DataArray:
        zsi = (da - da.mean(dim=dim)) / da.std(dim=dim)
        cs = np.power(zsi, 3) / da.sizes[dim]
        czi = (
            6.0 / cs * np.power((cs / 2.0 * zsi + 1.0), 1.0 / 3.0) - 6.0 / cs + cs / 6.0
        )
        return czi

    def fit(
        self,
        variable: str,
        time_period: str = "month",
        rolling_window: int = 3,
        modified: bool = False,
    ) -> None:

        print("Fitting China Z-Score Index")
        ds_window = rolling_cumsum(self.ds, rolling_window)

        if modified:  # modified china z index
            out_variable = "MCZI"
            func = self.MCZI
        else:
            out_variable = "CZI"
            func = self.CZI

        czi = apply_over_period(
            ds_window,
            func=func,
            in_variable=variable,
            out_variable=out_variable,
            time_str=time_period,
        )
        ds_window = ds_window.merge(czi.drop("month")).rename(
            {variable: f"{variable}_cumsum"}
        )

        self.index = ds_window
        print(f"Fitted China Z-Score Index and stored at `obj.index`")
