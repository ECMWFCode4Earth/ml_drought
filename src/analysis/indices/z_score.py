import xarray as xr

from .base import BaseIndices
from .utils import rolling_cumsum, apply_over_period


class ZScoreIndex(BaseIndices):
    """https://bit.ly/2xze9vg

    Calculation:
    -----------
    Calculated by subtracting the long term mean from an individual rainfall
    value and then dividing the difference by the standard deviation.

    ## References

    Akhtari, R., Morid, S., Mahdian, M.H., Smakhtin, V., 2009. Assessment of
    areal interpolation methods for spatial analysis of SPI and EDI drought
    indices. Int. J. Climatol. 29, 135–145.

    Dogan, S., Berktay, A., Singh, V.P., 2012. Comparison of multi-monthly
    rainfall-based drought severity indices, with application to semi-arid Konya
    closed basin, Turkey. J. Hydrol. 470–471, 255–268.

    Edwards, D.C., Mckee, T.B., 1997. Characteristics of 20th century drought in
    the United States at multiple time scales. Atmos. Sci. Pap. 63, 1–30.

    Komuscu, A.U., 1999. Using the SPI to analyze spatial and temporal patterns
    of drought in Turkey. Drought Network News (1994-2001). Paper 49. pp. 7–13.

    Morid, S., Smakhtin, V., Moghaddasi, M., 2006. Comparison of seven
    meteorological indices for drought monitoring in Iran. Int. J. Climatol. 26,
    971–985.

    Patel, N.R., Chopra, P., Dadhwal, V.K., 2007. Analyzing spatial patterns of
    meteorological drought using standardized precipitation index. Meteorol.
    Appl. 14, 329–336.

    Tsakiris, G., Vangelis, H., 2004. Towards a drought watch system based on
    spatial SPI. Water Resour. Manag. 18, 1–12.

    Wu, H., Hayes, M.J., Weiss, A., Hu, Q.I., 2001. An evaluation of the
    standardized precipitation index, the china-Zindex and the statistical
    Z-Score. Int. J. Climatol.21, 745–758. http://dx.doi.org/10.1002/joc.658.
    """

    name = "z_score_index"

    @staticmethod
    def ZSI(da: xr.DataArray, dim: str = "time", **kwargs) -> xr.DataArray:
        zsi = (da - da.median(dim=dim)) / da.std(dim=dim)
        return zsi

    def fit(
        self, variable: str, rolling_window: int = 3, time_str: str = "month"
    ) -> None:
        coords = [c for c in self.ds.coords]
        vars = [v for v in self.ds.variables if v not in coords]
        assert (
            variable in vars
        ), f"Must apply ZSI to a \
        variable in `self.ds`: {vars}"

        print(f"Fitting ZSI for variable: {variable}")
        # 1. calculate a cumsum over `rolling_window` timesteps
        ds_window = rolling_cumsum(self.ds, rolling_window)

        # 2. calculate the ZSI for the variable
        out_variable = "ZSI"
        zsi = apply_over_period(
            ds_window,
            func=self.ZSI,
            in_variable=variable,
            out_variable=out_variable,
            time_str=time_str,
        )
        ds_window = ds_window.merge(zsi.drop("month")).rename(
            {variable: f"{variable}_cumsum"}
        )

        self.index = ds_window
        print("Fitted ZSI and stored at `obj.index`")
