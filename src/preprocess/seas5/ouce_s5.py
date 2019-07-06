import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict


class OuceS5Data:
    hourly_s5_dir = Path('/lustre/soge1/data/incoming/seas5/1.0x1.0/6-hourly')
    daily_s5_dir = Path('/lustre/soge1/data/incoming/seas5/1.0x1.0/daily')

    @staticmethod
    def add_initialisation_date(ds: xr.Dataset, fname: Path) -> xr.Dataset:
    date_from_fname = pd.to_datetime(fname.stem.split('_')[-1], format='%Y%M')
    date_from_fname -= pd.to_timedelta('1 minute')
    return ds.expand_dims(
        {'initialisation_date': [date_from_fname]}
    )

    @staticmethod
    def create_forecast_horizon(ds: xr.Dataset, infer=False) -> xr.Dataset:
        if infer:
        # inferring the forecast_horizon dynamically doesn't work
        # because calculates the
            fh = pd.to_timedelta(ds.time.values - ds.initialisation_date.values)
        else:
            assert len(ds.time.values) == 860, f"The static forecast_horizon method\
            only works when the forecast_horizon lengths are the same"
            fh = FH
        ds['time'] = fh
        return ds.rename({'time': 'forecast_horizon'})

    @staticmethod
    def create_2D_time_coord(ds: xr.Dataset) -> xr.Dataset:
        time = ds.initialisation_date + ds.forecast_horizon
        return ds.assign_coords(time=time)

    @staticmethod
    def recreate_cds_s5(ds: xr.Dataset, fname: Path):
        """convert the preprocessed S5 data into format consistent with
        that downloaded from the CDS API for reproducibility.
        Required because of preprocessing done on OUCE server
        """
        # add `initialisation_date` from filename
        ds = add_initialisation_date(ds, fname)
        # convert `time` to `forecast_horizon`
        ds = create_forecast_horizon(ds)
        # create 2D `time` object (as in CDS API objects)
        ds = create_2D_time_coord(ds)
        return ds

    @staticmethod
    def read_ouce_s5_data(path: Path) -> xr.Dataset:
        """ Read and process OUCE S5 data into format consistent with CDS API """
        ds = xr.open_dataset(path)
        return recreate_cds_s5(ds, fname=path1)

    def get_ouce_filepaths(self, variable: str) -> List[Path]:
        """ For working on OUCE linux machine need to specify alternative
         path. We still want to write out the preprocessed data to
         the `self.data_dir / 'interim' / 'S5preprocessed'` but
         we need to read the data from a folder which we only have read
         permissions from.
        """
        target_folder = self.hourly_s5_dir
        valid_variables = [f.name for f in target_folder.glob('*')]
        assert variable in valid_variables, f"Invalid variable selected\
        We currently only have:{valid_variables}\
        You requested: {variable}"

        outfiles = list(target_folder.glob('**/*/*.nc'))
        outfiles.sort()
        return outfiles
