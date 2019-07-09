import xarray as xr
from pathlib import Path

from typing import Optional, Tuple
from enum import Enum

import climate_indices
from climate_indices import indices

from .base import BaseIndices


class SPI(BaseIndices):
    """https://climatedataguide.ucar.edu/climate-data/
    standardized-precipitation-index-spi
    """
    name = 'spi'

    # SPI should be monthly so hardcoding this to avoid confusion
    resample = True
    resample_str = 'month'

    def __init__(self, data_path: Path) -> None:
        super().__init__(data_path, resample_str=self.resample_str)

    @staticmethod
    def stack_pixels(da: xr.DataArray) -> xr.DataArray:
        return da.stack(point=('lat', 'lon')).groupby('point')

    def init_distribution(self, distribution: str) -> Enum:
        if distribution == 'gamma':
            dist = climate_indices.indices.Distribution.gamma
        elif distribution == 'pearson':
            dist = climate_indices.indices.Distribution.pearson
        else:
            assert False, f"{distribution} is not a valid distribution fit for SPI"

        self.distribution = distribution

        return dist

    def init_start_year(self, data_start_year: Optional[int] = None) -> int:
        if data_start_year is None:
            data_start_year = int(self.ds['time.year'].min().values)
            print(f"Setting the data_start_year automatically: {data_start_year}")

        assert isinstance(data_start_year, int), f"Expected data_start_year to be an integer"
        return data_start_year

    def init_calib_year(self, initial_yr: Optional[int],
                        final_yr: Optional[int]) -> Tuple[int, int]:
        if initial_yr is None:
            initial_yr = int(self.ds['time.year'].min().values)

            print(f"Setting the inital timeperiod for calibration manually:\n\
            inital year: {initial_yr}")

        if final_yr is None:
            final_yr = int(self.ds['time.year'].max().values)

            print(f"Setting the final timeperiod for calibration manually:\n\
            final year: {final_yr}")

        assert initial_yr >= int(self.ds['time.year'].min().values), f"intial_year\
        must be greater than the minimum year of the data\
        {initial_yr} >= {int(self.ds['time.year'].min().values)}"
        assert final_yr <= int(self.ds['time.year'].max().values), f"final_year\
        must be less than the maximum year of the data .\
        {final_yr} <= {int(self.ds['time.year'].max().values)}"

        return initial_yr, final_yr

    def init_periodicity(self, periodicity: Optional[str]) -> Enum:
        if periodicity is None:
            periodicity = 'gamma'

        if periodicity == 'monthly':
            period = climate_indices.compute.Periodicity.monthly
        elif periodicity == 'daily':
            period = climate_indices.compute.Periodicity.daily
        else:
            assert False, f"{periodicity} is not a valid periodicity for SPI"

        self.periodicity = periodicity

        return period

    def initialise_params(self,
                          scale: int,
                          distribution: str,
                          data_start_year: Optional[int],
                          calibration_year_initial: Optional[int],
                          calibration_year_final: Optional[int],
                          periodicity: Optional[str]
                          ) -> Tuple[int, Enum, int, int, int, int, Enum]:
        self.scale = scale
        # distribution must be a climate_indices enum type
        dist = self.init_distribution(distribution)
        self.data_start_year = self.init_start_year(data_start_year)
        self.calibration_year_initial, self.calibration_year_final = (
            self.init_calib_year(
                calibration_year_initial,
                calibration_year_final
            )
        )
        # period must be a climate_indices enum type
        period = self.init_periodicity(periodicity)

        return (
            self.scale, dist, self.data_start_year, self.calibration_year_initial,
            self.calibration_year_final, self.init_calib_year, period
        )

    def fit(self,
            variable: str,
            scale: int = 3,
            distribution: str = 'gamma',
            data_start_year: Optional[int] = None,
            calibration_year_initial: Optional[int] = None,
            calibration_year_final: Optional[int] = None,
            periodicity: Optional[str] = 'monthly',) -> None:
        """fit the index to self.ds writing to new self.index `xr.Dataset`

        Arguments:
        ---------
        variable: str
            the name of the variable in the self.ds (xr.Dataset) -> required to convert
            to a xr.DataArray for fitting the SPI

        scale: int = 3
            the window to calculate cumulative anomalies. Shorter term droughts will
            have smaller scales.

        distribution: str = 'gamma'
            the distribution used to fit to the raw precipitation data.
            {'gamma', 'pearson'}

        data_start_year: Optional[int] = None
            the starting year of the data series. Defaults to using the MINIMUM in the
            time series.
            **(we recommend leaving this parameter alone).

        calibration_year_initial: Optional[int] = None
            the first year of the calibration period (). Defaults to using the MINIMUM
            year in the time series.

        calibration_year_final: Optional[int] = None
            the final year of the calibration period. Defaults to the year before
            the final year. This would be changed to reflect that you don't want to include
            the year you're testing for how anomalous it was in your calibration period.

        periodicity: Optional[str] = 'monthly' -> None
            the periodicity of your data.
            {'monthly', 'daily'}

        """
        coords = [c for c in self.ds.coords]
        vars = [v for v in self.ds.variables if v not in coords]
        assert variable in vars, f"Must choose a variable from: {vars}"

        # stack the lat-lon pairs into pixels
        da = self.stack_pixels(self.ds[variable])

        # initialise the parameters (including enums from climate_indices)
        _, dist, _, _, _, _, period = self.initialise_params(
            scale, distribution, data_start_year, calibration_year_initial,
            calibration_year_final, periodicity
        )

        print(
            "\n---------------\n",
            "Fitting SPI index\n",
            f"distribution: {self.distribution}\n",
            f"data_start_year: {self.data_start_year}\n",
            f"calibration_year_initial: {self.calibration_year_initial}\n",
            f"calibration_year_final: {self.calibration_year_final}\n",
            f"periodicity: {self.periodicity}\n",
            "---------------\n",
        )

        index = xr.apply_ufunc(
            indices.spi, da, self.scale,
            dist, self.data_start_year, self.calibration_year_initial,
            self.calibration_year_final, period
        )

        self.index = index.unstack('point')

        print("Fitted")
