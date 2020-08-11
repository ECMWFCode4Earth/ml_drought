import xarray as xr
import pandas as pd
from pathlib import Path
from typing import List, Optional

from .fcast_horizon import FH


class OuceS5Data:
    """
    Example file:

    ```
    netcdf seas5_6-hourly_2m_temperature_200802 {
    dimensions:
        longitude = 360 ;
        latitude = 181 ;
        number = 25 ;
        time = 860 ;
    variables:
        float longitude(longitude) ;
            longitude:units = "degrees_east" ;
            longitude:long_name = "longitude" ;
        float latitude(latitude) ;
            latitude:units = "degrees_north" ;
            latitude:long_name = "latitude" ;
        int number(number) ;
            number:long_name = "ensemble_member" ;
        int time(time) ;
            time:units = "hours since 1900-01-01 00:00:0.0" ;
            time:long_name = "time" ;
            time:calendar = "gregorian" ;
        float t2m(time, number, latitude, longitude) ;
            t2m:_FillValue = 9.96921e+36f ;
            t2m:missing_value = 9.96921e+36f ;
            t2m:units = "K" ;
            t2m:long_name = "2 metre temperature" ;
    }
    ```
    """

    hourly_s5_dir = Path("/soge-home/data/model/seas5/1.0x1.0/6-hourly")
    daily_s5_dir = Path("/soge-home/data/model/seas5/1.0x1.0/daily")

    @staticmethod
    def add_initialisation_date(ds: xr.Dataset, fname: Path) -> xr.Dataset:
        date_from_fname = pd.to_datetime(fname.stem.split("_")[-1], format="%Y%m")
        return ds.expand_dims({"initialisation_date": [date_from_fname]})

    @staticmethod
    def create_forecast_horizon(ds: xr.Dataset, infer=False) -> xr.Dataset:
        if infer:
            # inferring the forecast_horizon dynamically doesn't work
            # because calculates the forecast horizons differently
            fh = pd.to_timedelta(ds.time.values - ds.initialisation_date.values)
        else:
            assert (
                len(ds.time.values) == 860
            ), f"The static forecast_horizon method\
            only works when the forecast_horizon lengths are the same"
            fh = FH
        ds["time"] = fh
        return ds.rename({"time": "forecast_horizon"})

    @staticmethod
    def create_2D_time_coord(ds: xr.Dataset) -> xr.Dataset:
        time = ds.initialisation_date + ds.forecast_horizon
        return ds.assign_coords(time=time)

    def recreate_cds_s5(self, ds: xr.Dataset, fname: Path, infer: bool) -> xr.Dataset:
        """convert the preprocessed S5 data into format consistent with
        that downloaded from the CDS API for reproducibility.
        Required because of preprocessing done on OUCE server
        """
        # add `initialisation_date` from filename
        ds = self.add_initialisation_date(ds, fname)
        # convert `time` to `forecast_horizon`
        ds = self.create_forecast_horizon(ds, infer)
        # create 2D `time` object (as in CDS API objects)
        ds = self.create_2D_time_coord(ds)
        return ds

    def read_ouce_s5_data(self, path: Path, infer: bool = False) -> xr.Dataset:
        """ Read and process OUCE S5 data into format consistent with CDS API """
        ds = xr.open_dataset(path)
        return self.recreate_cds_s5(ds, fname=path, infer=infer)

    def get_ouce_filepaths(
        self, variable: str, parent_dir: Optional[Path] = None
    ) -> List[Path]:
        """ For working on OUCE linux machine need to specify alternative
         path. We still want to write out the preprocessed data to
         the `self.data_dir / 'interim' / 'S5preprocessed'` but
         we need to read the data from a folder which we only have read
         permissions from.
        """
        target_folder = self.hourly_s5_dir if parent_dir is None else parent_dir
        valid_variables = [f.name for f in target_folder.glob("*")]
        assert (
            variable in valid_variables
        ), f"Invalid variable selected\
        We currently only have:{valid_variables}\
        You requested: {variable}"

        outfiles = list(target_folder.glob("**/*/*.nc"))
        outfiles.sort()
        return outfiles
