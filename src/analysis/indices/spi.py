import xarray as xr
from pathlib import Path

from typing import Optional, Tuple
from enum import Enum

from .base import BaseIndices

climate_indices = None
indices = None


class SPI(BaseIndices):
    """https://climatedataguide.ucar.edu/climate-data/
    standardized-precipitation-index-spi

    Number of standard deviations that the observed value would deviate from
    the long-term mean, for a normally distributed random variable.

    Comparing the precipitation total for the chosen interval against a
    cumulative probability distribution for the precipitation data (for the
    identical interval). For example, what is statistical interpretation of the
    one-month precipitation total (e.g., 29 mm), compared to all known
    one -month totals? It is necessary to view the drought according to
    the climatological norms for the location and season.

    Calculation:
    -----------
    a long-term time series of precipitation accumulations over the desired
    time scale are used to estimate an appropriate probability density
    function.
    Since precipitation is not normally distributed, a transformation is first
    applied so that the transformed precipitation values follow a normal
    distribution.
    This normal distribution can then be used to understand the divergence
    from normal (mean) conditions in terms of standard deviation / probability.

    ## References
    World Meteorological Organization, (2012) Standardized Precipitation Index
    User Guide (M. Svoboda, M. Hayes and D. Wood). (WMO-No. 1090), Geneva.

    Guttman, N. B., 1999: Accepting the Standardized Precipitation Index:
    A calculation algorithm. J. Amer. Water Resour. Assoc.., 35(2), 311-322.

    McKee, T. B., N. J. Doesken, and J. Kliest, 1993: The relationship of
    drought frequency and duration to time scales. In Proceedings of the 8th
    Conference of Applied Climatology, 17-22 January, Anaheim, CA. American
    Meteorological Society, Boston, MA. 179-184.

    Vicente-Serrano, Sergio M., Santiago Beguería, Juan I. López-Moreno, 2010:
    A Multiscalar Drought Index Sensitive to Global Warming: The Standardized
    Precipitation Evapotranspiration Index. J. Climate, 23, 1696–1718.
    """

    name = "spi"

    # SPI should be monthly so hardcoding this to avoid confusion
    resample = True
    resample_str = "month"

    def __init__(
        self, file_path: Optional[Path] = None, ds: Optional[xr.Dataset] = None
    ) -> None:
        super().__init__(file_path, ds=ds, resample_str=self.resample_str)

        global indices
        global climate_indices

        if climate_indices is None:
            import climate_indices
        if indices is None:
            from climate_indices import indices

    @staticmethod
    def stack_pixels(da: xr.DataArray) -> xr.DataArray:
        return da.stack(point=("lat", "lon")).groupby("point")

    def init_distribution(self, distribution: str) -> Enum:
        if distribution == "gamma":
            dist = climate_indices.indices.Distribution.gamma  # type: ignore
        elif distribution == "pearson":
            dist = climate_indices.indices.Distribution.pearson  # type: ignore
        else:
            assert False, f"{distribution} is not a valid distribution fit for SPI"

        self.distribution = distribution

        return dist

    def init_start_year(self, data_start_year: Optional[int] = None) -> int:
        if data_start_year is None:
            data_start_year = int(self.ds["time.year"].min().values)
            print(f"Setting the data_start_year automatically: {data_start_year}")

        assert isinstance(
            data_start_year, int
        ), f"Expected data_start_year to be an integer"
        return data_start_year

    def init_calib_year(
        self, initial_yr: Optional[int], final_yr: Optional[int]
    ) -> Tuple[int, int]:
        if initial_yr is None:
            initial_yr = int(self.ds["time.year"].min().values)

            print(
                f"Setting the inital timeperiod for calibration manually:\n\
            inital year: {initial_yr}"
            )

        if final_yr is None:
            final_yr = int(self.ds["time.year"].max().values)

            print(
                f"Setting the final timeperiod for calibration manually:\n\
            final year: {final_yr}"
            )

        assert initial_yr >= int(
            self.ds["time.year"].min().values
        ), f"intial_year\
        must be greater than the minimum year of the data\
        {initial_yr} >= {int(self.ds['time.year'].min().values)}"
        assert final_yr <= int(
            self.ds["time.year"].max().values
        ), f"final_year\
        must be less than the maximum year of the data .\
        {final_yr} <= {int(self.ds['time.year'].max().values)}"

        return initial_yr, final_yr

    def init_periodicity(self, periodicity: Optional[str]) -> Enum:
        if periodicity is None:
            periodicity = "gamma"

        if periodicity == "monthly":
            period = climate_indices.compute.Periodicity.monthly  # type: ignore
        elif periodicity == "daily":
            period = climate_indices.compute.Periodicity.daily  # type: ignore
        else:
            assert False, f"{periodicity} is not a valid periodicity for SPI"

        self.periodicity = periodicity

        return period

    def initialise_params(
        self,
        scale: int,
        distribution: str,
        data_start_year: Optional[int],
        calibration_year_initial: Optional[int],
        calibration_year_final: Optional[int],
        periodicity: Optional[str],
    ) -> Tuple[int, Enum, int, int, int, int, Enum]:
        self.scale = scale
        # distribution must be a climate_indices enum type
        dist = self.init_distribution(distribution)
        self.data_start_year = self.init_start_year(data_start_year)
        (
            self.calibration_year_initial,
            self.calibration_year_final,
        ) = self.init_calib_year(calibration_year_initial, calibration_year_final)
        # period must be a climate_indices enum type
        period = self.init_periodicity(periodicity)

        return (  # type: ignore
            self.scale,
            dist,
            self.data_start_year,
            self.calibration_year_initial,
            self.calibration_year_final,
            self.init_calib_year,
            period,
        )

    def fit(
        self,
        variable: str,
        scale: int = 3,
        distribution: str = "gamma",
        data_start_year: Optional[int] = None,
        calibration_year_initial: Optional[int] = None,
        calibration_year_final: Optional[int] = None,
        periodicity: Optional[str] = "monthly",
    ) -> None:
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
            scale,
            distribution,
            data_start_year,
            calibration_year_initial,
            calibration_year_final,
            periodicity,
        )

        print(
            "\n---------------\n",
            f"Fitting SPI{scale} index\n",
            f"distribution: {self.distribution}\n",
            f"data_start_year: {self.data_start_year}\n",
            f"calibration_year_initial: {self.calibration_year_initial}\n",
            f"calibration_year_final: {self.calibration_year_final}\n",
            f"periodicity: {self.periodicity}\n",
            "---------------\n",
        )

        try:
            index = xr.apply_ufunc(  # type: ignore
                indices.spi,  # type: ignore
                da,
                self.scale,
                dist,
                self.data_start_year,
                self.calibration_year_initial,
                self.calibration_year_final,
                period,
            )
        except ValueError as E:
            print("** ValueError: ")
            print(E)
            print("\the current fix is rather nasty edit of xarray")
            print(
                "the indices.spi computation sometimes collapses the dimensionality"
                " of the groupby object. "
                "Therefore added some code to L578 "
                "~/miniconda3/envs/crop/lib/python3.7/site-packages/xarray/core/computation.py)"
                "\n> # TODO: TOMMY ADDED\n"
                "\nif (data.ndim == 1) and (len(dims) == 2):"
                "\n\tdata = np.expand_dims(data, -1)"
                "\n"
            )
            print(
                "I am pretty sure that this is probably something"
                " that we need to fix from the indices code"
            )

        self.index = index.unstack("point").to_dataset(name=f"SPI{scale}")

        print("Fitted SPI and stored at `obj.index`")
