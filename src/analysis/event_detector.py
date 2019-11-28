import xarray as xr
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Any
import warnings
from ..utils import get_ds_mask, create_shape_aligned_climatology

rle = None
longest_run = None


class EventDetector:
    """A flexible method for detecting events and calculating the size of runs
    in a timeseries of interest.

    an EventDetector object calculates the climatology and threshold values
    (per pixel) and then can compute another xr.Dataset of `runs` which
    are consecutive periods of threshold exceedence (above/below).

    >>> path_to_data = Path('data/interim/chirps_preprocessed/chirps_kenya.nc')
    >>> e = EventDetector(path_to_data)
    >>> e.detect(variable='precip', method='std', time_period='month', hilo='low')
    >>> e.calculate_runs()
    """

    def __init__(self, path_to_data: Path) -> None:

        global rle
        global longest_run
        if rle is None:
            from xclim.run_length import rle
        if longest_run is None:
            from xclim.run_length import longest_run

        assert (
            path_to_data.exists()
        ), f"{path_to_data} does not point to an existing file!"
        self.path_to_data = path_to_data
        self.ds = self.read_data(path_to_data)

    def read_data(self, path_to_data: Path) -> xr.Dataset:

        try:
            ds = xr.open_dataset(path_to_data)

        except ValueError:
            print(ValueError)
            warnings.warn("Having to decode_times=False because unsupported calendar")
            warnings.warn("You will need to manually encode your timesteps in self.ds")
            ds = xr.open_dataset(path_to_data, decode_times=False)

        print(f"{path_to_data.name} read!")
        ds = ds.sortby("time")
        # TODO: infer time frequency ?
        return ds

    @staticmethod
    def calculate_threshold(
        ds: xr.Dataset,
        clim: xr.Dataset,
        method: str,
        time_period: str,
        variable: str,
        hilo: Optional[str] = None,
        value: Optional[float] = None,
    ) -> xr.DataArray:
        """Calculate the threshold using the method defined in `method`.
        This calculates a threshold value above/below which a value is
        defined as anomalous and therefore, set to true. Compared against
        a reference climatology in methods ["q90","q10","std"]. Compared
        against an absolute value in method ["abs"]

        Arguments:
        ----------
        ds: xr.Dataset
            dataset that you want to calculate the threshold for

        clim: xr.Dataset
            the climatology dataset

        method: str
            the method used to create the threshold values
            {"q90","q10","std","abs",}

        time_period: str
            the time period to groupby
            {'daysofyear', 'month', 'year', 'season'}

        variable: str
            the name of the variable in `ds`

        hilo: Optional[str] = None
            Whether we are calculating the value above or below the threshold.
            REQUIRED for `method` == 'std'
            {'high', 'low'}

        value: Optional[float] = None
            if the method is `abs` then also need to provide a value to
            be used as the threshold value.
        """
        if method == "q90":
            warnings.warn(f"this method ({method}) is currently super slow")
            thresh = (
                ds.groupby(f"time.{time_period}")
                # .quantile(q=0.9)  # updated v0.12.2
                .reduce(np.nanpercentile, dim="time", q=90)
            )

        elif method == "q10":
            warnings.warn(f"this method ({method}) is currently super slow")
            thresh = (
                ds.groupby(f"time.{time_period}").reduce(
                    np.nanpercentile, dim="time", q=10
                )
                # .quantile(q=0.1)
            )

        elif method == "std":
            assert (
                hilo is not None
            ), f"If you want to calculate the threshold as std \
                from mean, then have to specify `hilo` = ['low','high']"
            std = ds.groupby(f"time.{time_period}").std(dim="time")
            thresh = clim - std if hilo == "low" else clim + std

        elif method == "abs":
            values = np.ones(ds[variable].shape) * value
            thresh = xr.Dataset(
                {variable: (["time", "lat", "lon"], values)},
                coords={"lat": ds.lat, "lon": ds.lon, "time": ds.time},
            )
            thresh = thresh.groupby(f"time.{time_period}").first()

        else:
            assert (
                False
            ), 'Only implemented threshold calculations for \
                ["q90","q10","std","abs",]'

        return thresh

    def get_thresh_clim_dataarrays(
        self,
        ds: xr.Dataset,
        time_period: str,
        variable: str,
        hilo: str = None,
        method: str = "std",
        value: Optional[float] = None,
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        """Get the climatology and threshold xarray objects """
        # compute climatology (`mean` over `time_period`)
        clim = ds.groupby(f"time.{time_period}").mean(dim="time")
        print(f"Calculated climatology (mean for each {time_period}) - `clim`")
        # compute the threshold value based on `method`
        thresh = self.calculate_threshold(
            ds,
            clim,
            method=method,
            time_period=time_period,
            hilo=hilo,
            variable=variable,
            value=value,
        )
        print("Calculated threshold - `thresh`")
        return clim, thresh

    def calculate_threshold_exceedences(
        self,
        variable: str,
        time_period: str,
        hilo: str,
        method: str = "std",
        value: Optional[float] = None,
    ) -> Tuple[Any, Any, Any]:
        """Flag the pixel-times that exceed a threshold defined via
        the `method` argument.
        """
        ds = self.ds

        # calculate climatology and threshold
        clim, thresh = self.get_thresh_clim_dataarrays(
            ds, time_period, hilo=hilo, method=method, variable=variable, value=value
        )

        # assign objects to object here because the copying
        #  is essential for processing but makes it confusing to
        #  understand
        self.clim = clim
        self.thresh = thresh

        # make them the same size as the ds variable
        clim_ext = create_shape_aligned_climatology(ds, clim, variable, time_period)
        thresh_ext = create_shape_aligned_climatology(ds, thresh, variable, time_period)
        # TODO: do I want to store these extended datasets?

        # flag exceedences
        if hilo == "low":
            exceed = (ds < thresh_ext)[variable]
        elif hilo == "high":
            exceed = (ds > thresh_ext)[variable]

        return clim_ext, thresh_ext, exceed

    def detect(
        self,
        variable: str,
        time_period: str,
        hilo: str,
        method: str = "std",
        value: Optional[float] = None,
    ) -> None:
        """
        Detect threshold exceedences above or below a threshold. The threshold
        can also be calculated or applied from elsewhere.

        Arguments:
        ---------
        variable: str
            name of the variable that you are comparing to the climatology

        time_period: str
            the period string used to calculate the climatology and define
             the time
             time_period = {'dayofyear', 'season', 'month'}

        hilo: str
            are you looking for exceedences above a threshold = 'high'
             or below a threshold = 'low'.
             Valid values: {'high','low'}.

        method: str
            How you want to calculate the threshold to flag exceedences
             Valid values: {'q90', 'q10', 'std', 'abs',}

        """
        print(
            f"Detecting {variable} exceedences ({hilo}) of threshold:\
         {method}. The threshold is unique for each {time_period}"
        )
        self.variable = variable
        _, _, exceed = self.calculate_threshold_exceedences(
            variable, time_period, hilo, method=method, value=value
        )

        # self.exceedences = self.reapply_mask_to_boolean_xarray(variable, exceed)
        self.exceedences = exceed
        print(f"** exceedences calculated **")

    def reapply_mask_to_boolean_xarray(
        self, variable_of_mask: str, da: xr.DataArray
    ) -> xr.DataArray:
        """Because boolean comparisons in xarray return False for nan values
        we need to reapply the mask from the original `da` to mask out the sea
        or invalid values (for example).

        Arguments:
        ---------
        variable_of_mask: str
            the variable that you want to use in `self.ds` as the mask.
            The `np.nan` values in `self.ds[variable]` will be marked as `True`

        da: xr.DataArray
            the boolean DataArray (for example `self.exceedences`) to reapply
            the mask to

        Returns:
        -------
        xr.DataArray with dtype of `int`, because `bool` dtype doesn't store
        masks / nan values very well.

        NOTE:
            1. Uses the input dataset for the mask - TODO: does this make
             sense?
        """
        assert da.dtype == np.dtype(
            "bool"
        ), f"This function \
        currrently works on boolean xr.Dataset objects only"

        mask = get_ds_mask(self.ds[variable_of_mask])
        da = da.astype(int).where(~mask)

        return da

    def calculate_runs(self) -> xr.Dataset:
        """use xclim to calculate the number of consecutive exceedences.
        For each pixel-time calculate the `run` of exceedences at the
        current timestep.

        e.g.
            [True, True, False, True, False, True, True, True] =>
            [1, 2, 0, 1, 0, 1, 2, 3]

            TODO: want to make this general enough to work with subset data too
        """
        assert self.exceedences.dtype == np.dtype(
            "bool"
        ), f"\
            Expected exceedences to be an array of boolean type.\
             Got {self.exceedences.dtype}"

        runs = rle(self.exceedences).load()  # type: ignore

        # apply the same mask as TIME=0 (e.g. for the sea-land mask)
        mask = get_ds_mask(self.ds[self.variable])
        runs = runs.where(~mask)

        return runs

    def calculate_longest_run(self, resample_str: Optional[str] = None) -> xr.Dataset:
        """ Calculate the longest run in the dataset
        TODO: fix this argument to work with other resample_str """
        return longest_run(self.exceedences, dim="time").load()  # type: ignore
