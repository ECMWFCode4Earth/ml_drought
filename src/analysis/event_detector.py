import xarray as xr
import numpy as np
from pathlib import Path
from xclim.run_length import rle, longest_run  # , windowed_run_events
from typing import Tuple, Optional
import warnings

from scripts.eng_utils import get_ds_mask


class EventDetector():
    def __init__(self,
                 path_to_data: Path,
                 data_folder: Path = Path('data')) -> None:
        self.data_folder = data_folder
        self.interim_folder = data_folder / "interim"
        self.processed_folder = data_folder / "processed"

        if not self.processed_folder.exists():
            self.processed_folder.mkdir()

        assert path_to_data.exists(), f"{path_to_data} does not point to an existing file!"
        self.ds = self.read_data(path_to_data)

    def read_data(self, path_to_data: Path) -> xr.Dataset:
        ds = xr.open_dataset(path_to_data)
        print(f"{path_to_data.name} read!")
        ds = ds.sortby('time')
        # TODO: infer time frequency ?
        return ds

    @staticmethod
    def calculate_threshold(ds: xr.Dataset,
                            clim: xr.Dataset,
                            method: str,
                            time_period: str,
                            variable: str,
                            hilo: Optional[str] = None,
                            value: Optional[int] = None,) -> xr.DataArray:
        """Calculate the threshold based on the `method` argument
        method: str
            ["q90","q10","std","abs",]
        """
        if method == "q90":
            warnings.warn('this method is currently super slow')
            thresh = ds.groupby(f'time.{time_period}').reduce(np.nanpercentile, dim='time', q=0.9)

        elif method == "q10":
            warnings.warn('this method is currently super slow')
            thresh = ds.groupby(f'time.{time_period}').reduce(np.nanpercentile, dim='time', q=0.1)

        elif method == "std":
            assert hilo is not None, f"If you want to calculate the threshold as std \
                from mean, then have to specify `hilo` = ['low','high']"
            std = ds.groupby(f'time.{time_period}').std(dim='time')
            thresh = clim - std if hilo == 'low' else clim + std

        elif method == "abs":
            assert False, "Not yet implemented the absolute value threshold"
            values = np.ones(ds[variable].shape) * value
            thresh = xr.Dataset(
                {variable: (['time', 'latitude', 'longitude'], values)},
                coords={
                    'latitude': ds.latitude,
                    'longitude': ds.longitude,
                    'time': ds.time,
                }
            )

        else:
            assert False, 'Only implemented threshold calculations for \
                ["q90","q10","std","abs",]'

        return thresh

    def get_thresh_clim_dataarrays(self,
                                   ds: xr.Dataset,
                                   time_period: str,
                                   variable: str,
                                   hilo: str = None,
                                   method: str = 'std') -> Tuple[xr.Dataset, xr.Dataset]:
        """Get the climatology and threshold xarray objects """
        # compute climatology (`mean` over `time_period`)
        clim = ds.groupby(f'time.{time_period}').mean(dim='time')
        # compute the threshold value based on `method`
        thresh = self.calculate_threshold(
            ds, clim, method=method, time_period=time_period,
            hilo=hilo, variable=variable
        )
        return clim, thresh

    @staticmethod
    def create_shape_aligned_climatology(ds, clim, variable, time_period):
        """match the time dimension of `clim` to the shape of `ds` so that can
        perform simple calculations / arithmetic on the values of clim

        Arguments:
        ---------
        ds : xr.Dataset
            the dataset with the raw values that you are comparing to climatology

        clim: xr.Dataset
            the climatology values for a given `time_period`

        variable: str
            name of the variable that you are comparing to the climatology

        time_period: str
            the period string used to calculate the climatology
             time_period = {'dayofyear', 'season', 'month'}

        Notes:
            1. assumes that `latitude` and `longitude` are the
            coord names in ds

        """
        ds[time_period] = ds[f'time.{time_period}']

        values = clim[variable].values
        keys = clim[time_period].values
        # map the `time_period` to the `values` of the climatology (threshold or mean)
        lookup_dict = dict(zip(keys, values))

        # extract an array of the `time_period` values from the `ds`
        timevals = ds[time_period].values

        # use the lat lon arrays (climatology values) in `lookup_dict` corresponding
        #  to time_period values defined in `timevals` and stack into new array
        new_clim_vals = np.stack([lookup_dict[timevals[i]] for i in range(len(timevals))])

        assert new_clim_vals.shape == ds[variable].shape, f"\
            Shapes for new_clim_vals and ds must match! \
             new_clim_vals.shape: {new_clim_vals.shape} \
             ds.shape: {ds[variable].shape}"

        # copy that forward in time
        new_clim = xr.Dataset(
            {variable: (['time', 'latitude', 'longitude'], new_clim_vals)},
            coords={
                'latitude': clim.latitude,
                'longitude': clim.longitude,
                'time': ds.time,
            }
        )

        return new_clim

    def calculate_threshold_exceedences(self,
                                        variable: str,
                                        time_period: str,
                                        hilo: str,
                                        method: str = 'std') -> Tuple[xr.Dataset]:
        """Flag the pixel-times that exceed a threshold defined via
        the `method` argument.
        """
        ds = self.ds

        # calculate climatology and threshold
        clim, thresh = self.get_thresh_clim_dataarrays(
            ds, time_period, hilo=hilo, method=method,
            variable=variable
        )

        # assign objects to object here because the copying
        #  is essential for processing but makes it confusing to
        #  understand
        self.clim = clim
        self.thresh = thresh

        # make them the same size as the ds variable
        clim_ext = self.create_shape_aligned_climatology(
            ds, clim, variable, time_period
        )
        thresh_ext = self.create_shape_aligned_climatology(
            ds, thresh, variable, time_period
        )
        # TODO: do I want to store these extended datasets?

        # flag exceedences
        if hilo == 'low':
            exceed = (ds < thresh_ext)[variable]
        elif hilo == 'high':
            exceed = (ds > thresh_ext)[variable]

        return clim_ext, thresh_ext, exceed

    def detect(self,
               variable: str,
               time_period: str,
               hilo: str,
               method: str = 'std') -> None:
        """

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
        _, _, exceed = self.calculate_threshold_exceedences(
            variable, time_period, hilo, method=method
        )
        self.exceedences = exceed

    @staticmethod
    def reapply_mask_to_boolean_xarray(self, variable: str) -> xr.Dataset:
        """Because boolean comparisons in xarray return False for nan values
        we need to reapply the mask from the original `ds` to mask out the sea
        or invalid values (for example).

        NOTE:
        """
        mask = get_ds_mask(self.ds, variable)

        return mask

    def calculate_runs(self) -> xr.Dataset:
        """use xclim to calculate the number of consecutive exceedences.
        For each pixel-time calculate the `run` of exceedences at the
        current timestep.

        e.g.
            [True, True, False, True, False, True, True, True] =>
            [1, 2, 0, 1, 0, 1, 2, 3]


        """
        assert self.exceedences.dtype == np.dtype('bool'), f"\
            Expected exceedences to be an array of boolean type.\
             Got {self.exceedences.dtype}"

        runs = rle(self.exceedences).load()
        return runs

    def calculate_longest_run(self,
                              resample_str: Optional[str] = None) -> xr.Dataset:
        """ """
        longest_run(self.exceedences, dim='time').load()
        return
