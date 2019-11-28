import xarray as xr
import numpy as np

from .base import BaseIndices
from .utils import rolling_cumsum, apply_over_period


class AnomalyIndex(BaseIndices):
    """ The Rainfall Anomaly Index

    Calculation:
    -----------
    incorporates a ranking procedure to assign magnitudes to
     positive and negative anomalies.
    two phases, positive precipitation anomalies and
     negative precipitation anomalies.

    Notes:
    -----
    Variations within the year need to be small compared to
     temporal variations.
    Calculating RAI requires a serially complete dataset with estimates of missing values.

    ## References

    Rooy, M.P.V., 1965. A rainfall anomaly index independent of time and space.
     Weather Bureau of South Africa 14, 43-48.

    Freitas, M.A.S., 2004. A Previsão de Secas e a Gestão Hidroenergética: O
     Caso da Bacia do Rio Parnaíba no Nordeste do Brasil. Seminário
      Internacional sobre Represas y Operación de Embalses, Puerto Iguazú.

    Kraus, E.B., 1977: Subtropical droughts and cross-equatorial energy
     transports. Monthly Weather Review, 105(8): 1009-1018. DOI:
     10.1175/1520-0493(1977)105<1009:SDACEE>2.0.CO;2.

    van Rooy, M.P., 1965: A Rainfall Anomaly Index independent of time and
     space. Notos, 14: 43–48.

    http://www.droughtmanagement.info/rainfall-anomaly-index-rai/
    """

    name = "rainfall_anomaly_index"

    @staticmethod
    def assign_magnitudes(
        y: xr.Dataset,
        anom: xr.DataArray,
        rai_plus: xr.DataArray,
        rai_minus: xr.DataArray,
        variable: str,
    ) -> xr.Dataset:
        """ assign magnitudes to positive and negative anomalies """
        # REVERSE boolean indexing
        # if anom values are NOT less than/equal than fill with PLUS values
        y_plus = y.where(~(anom >= 0), other=rai_plus)
        y_plus_mask = y.where(~(anom >= 0)).isnull()
        y_plus = y_plus.where(y_plus_mask)

        y_minus = y.where(~(anom < 0), other=rai_minus)
        y_minus_mask = y.where(~(anom < 0)).isnull()
        y_minus = y_minus.where(y_minus_mask)

        # convert to numpy masked array
        y_plus_m_np = y_plus_mask[variable].values
        y_plus_np = y_plus[variable].values
        y_plus_np = np.ma.array(y_plus_np, mask=(~y_plus_m_np))

        y_minus_m_np = y_minus_mask[variable].values
        y_minus_np = y_minus[variable].values
        y_minus_np = np.ma.array(y_minus_np, mask=(~y_minus_m_np))

        # recombine masked arrays
        rai_array = np.ma.array(
            y_plus_np.filled(1) * y_minus_np.filled(1),
            mask=(y_plus_np.mask * y_minus_np.mask),
        )

        rai = xr.ones_like(y)
        rai[variable] = (["time", "lat", "lon"], rai_array)
        return rai

    def RAI(self, da: xr.DataArray, variable: str, dim: str = "time") -> xr.DataArray:
        """ Rainfall Anomaly Index """
        # calculations
        y = da.copy()
        x1 = da.copy().sortby(dim, ascending=False)  # high -> low over TIME

        # sample average, max average, min average
        x_avg = x1.mean(dim=dim)  # monthly average over all time
        mx_avg = x1.isel(time=slice(0, 10)).mean(dim=dim)  # max mean (top 10)
        mn_avg = x1.isel(time=slice(-11, -1)).mean(dim=dim)  # min mean (bottom 10) NB

        # anomaly = value - mean
        anom = da - x_avg

        # where anomaly is >=0, get the ratio to difference of max_avg - x_avg
        rai_plus = 3.0 * anom.where(anom >= 0) / (mx_avg - x_avg)
        rai_minus = -3.0 * anom.where(anom < 0) / (mn_avg - x_avg)

        # assign the magnitudes for positive and negative anomalies
        # convert to numpy masked arrays and then back out to xarray
        rai = self.assign_magnitudes(y, anom, rai_plus, rai_minus, variable)

        return rai

    def fit(
        self, variable: str, time_period: str = "month", rolling_window: int = 3
    ) -> None:

        print("Fitting Rainfall Anomaly Index")

        # 1. calculate a cumsum over `rolling_window` timesteps
        ds_window = rolling_cumsum(self.ds, rolling_window)

        out_variable = "RAI"

        rai = apply_over_period(  # type: ignore
            ds_window,
            func=self.RAI,
            in_variable=variable,
            out_variable=out_variable,
            time_str=time_period,
            **{"variable": variable},
        )
        rai = rai.drop("month")
        ds_window = ds_window.merge(rai).rename({variable: f"{variable}_cumsum"})

        self.index = ds_window
        print(f"Fitted Rainfall Anomaly Index and stored at `obj.index`")
