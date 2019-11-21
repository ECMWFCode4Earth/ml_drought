import xarray as xr
from typing import List, Optional

from .base import BaseIndices
from .utils import rolling_cumsum, create_shape_aligned_climatology


class PercentNormalIndex(BaseIndices):
    """
    https://bit.ly/2XAYXIK

    Calculation:
    -----------
    calculated by simply dividing actual precipitation by normal (30-year mean)
     precipitation and multiplying the result by 100.

    This approach of PNI (McKee et al. 1993) has value in its simplicity and
    transparency, especially because all sectors tend to “know what it means,
    ” and it is noteworthy that this approach has strong support in countries
    such as Indonesia. A downside of this approach is that it does not
    necessarily detect the extremes in drought conditions, and this can be a
    problem in very arid areas.

    This approach also requires a good knowledge of local conditions to make it
    useful. Hayes (2000 and 2006) suggests analyses using percent of normal are
    most effective when used for a single region or a single season.
    Conversely, percent of normal may be misunderstood and provide different
    indications of conditions, depending on the location and season.

    Hayes (2000) points out that one of the disadvantages of using the percent
    of normal precipitation is that the mean, or average, precipitation may
    differ considerably from the median precipitation (which is the value
    exceeded by 50% of the precipitation occurrences in a long-term climate
    record) in many world regions, especially those with high year-to-year
    rainfall variability. Thus, use of the percent of normal comparison
    requires a normal distribution in rainfall, where the mean and median are
    considered to be the same.

    ## References

    Hayes, M.J. 2000. Drought indices. National Drought Mitigation Center,
    University of Nebraska, Lincoln, Nebraska.

    Hayes, M.J., 2006: Drought Indices. Van Nostrand’s Scientific Encyclopedia,
    John Wiley & Sons, Inc. DOI: 10.1002/0471743984.vse8593.

    McKee, T.B., N.J. Doesken, and J. Kieist. 1993. The relationship of drought
    frequency and duration of time scales. Pages 179-184 in Proceedings of the
    8th Conference on Applied Climatology. Anaheim, California. American
    Meteorological Society, Boston.

    Morid, S., Smakhtin, V., Moghaddasi, M., 2006. Comparison of seven
    meteorological indices for drought monitoring in Iran. Int. J. Climatol.
    26, 971–985.
    """

    name = "percent_normal_index"

    @staticmethod
    def calculatePNI(
        ds: xr.Dataset,
        variable: str,
        time_period: str,
        rolling_window: int = 3,
        clim_period: Optional[List[str]] = None,
    ) -> xr.DataArray:
        """calculate Percent of Normal Index (PNI)

            Arguments:
            ---------
            ds : xr.Dataset
                the dataset with the raw values that you are comparing to climatology

            variable: str
                name of the variable that you are comparing to the climatology

            time_period: str
                the period string used to calculate the climatology
                 time_period = {'dayofyear', 'season', 'month'}

            rolling_window: int Default = 3
                the size of the cumsum window (in timesteps)

        """
        # calculate the rolling window (cumsum over time)
        ds_window = rolling_cumsum(ds, rolling_window)

        # calculate climatology based on time_period
        mthly_climatology = ds_window.groupby(f"time.{time_period}").mean(dim="time")
        clim = create_shape_aligned_climatology(
            ds_window, mthly_climatology, variable, time_period
        )

        # calculate the PNI
        PNI = (ds_window / clim) * 100
        # drop the initial nans caused by the widnowed cumsum
        #  (e.g. window=3 the first 2 months)
        PNI = (
            PNI.dropna(dim="time", how="all")
            .rename({variable: "PNI"})
            .merge(ds_window)
            .rename({variable: f"{variable}_cumsum"})
        )

        return PNI, clim

    def fit(
        self, variable: str, rolling_window: int = 3, time_str: str = "month"
    ) -> None:

        coords = [c for c in self.ds.coords]
        vars = [v for v in self.ds.variables if v not in coords]
        assert (
            variable in vars
        ), f"Must apply PNI to a \
        variable in `self.ds`: {vars}"

        print(f"Fitting PNI for variable: {variable}")
        PNI, clim = self.calculatePNI(
            self.ds,
            variable=variable,
            time_period=time_str,
            rolling_window=rolling_window,
            clim_period=None,
        )

        self.index = PNI
        self.climatology = clim

        print("Fitted PNI and stored at `obj.index`")
