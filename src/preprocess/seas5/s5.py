import numpy as np
from pathlib import Path
import xarray as xr
from functools import partial
import multiprocessing
from shutil import rmtree
from typing import Optional, List, Tuple, cast

from ..base import BasePreProcessor
from .ouce_s5 import OuceS5Data


class S5Preprocessor(BasePreProcessor):
    dataset: str = 's5'

    def __init__(self, data_folder: Path = Path('data'),
                 ouce_server: bool = False,
                 parallel: bool = False) -> None:
        super().__init__(data_folder)
        self.ouce_server = ouce_server
        self.parallel = parallel

    def get_filepaths(self, folder: str = 'raw') -> List[Path]:
        """ because reading .grib files have to rewrite get_filepaths"""
        if folder == 'raw':
            target_folder = self.raw_folder / self.dataset
        else:
            target_folder = self.interim
        outfiles = list(target_folder.glob('**/*.grib'))
        outfiles.sort()
        return outfiles

    @staticmethod
    def read_grib_file(filepath: Path) -> xr.Dataset:
        assert filepath.suffix in ['.grib', '.grb'], f"This method is for \
        `grib` files. Not for {filepath.as_posix()}"
        ds = xr.open_dataset(filepath, engine='cfgrib')

        ds = ds.rename({
            'time': 'initialisation_date', 'step': 'forecast_horizon',
            'valid_time': 'time'
        })
        if ds.surface.values.size == 1:
            ds = ds.drop('surface')

        return ds

    @staticmethod
    def create_filename(filepath: Path,
                        output_dir: Path,
                        variable: str,
                        regrid: Optional[xr.Dataset] = None,
                        subset_name: Optional[str] = None) -> Path:
        # TODO: do we want each variable in separate folders / .nc files?
        subset_name = ('_' + subset_name) if subset_name is not None else ''
        filename = filepath.stem + f'_{variable}{subset_name}.nc'
        output_path = output_dir / filename
        return output_path

    def _preprocess(self,
                    filepath: Path,
                    subset_str: Optional[str] = None,
                    regrid: Optional[xr.Dataset] = None) -> Tuple[Path, str]:
        """preprocess a single s5 dataset (multi-variables per `.nc` file)"""
        print(f"\nWorking on {filepath.name}")

        if self.ouce_server:
            # undoes the preprocessing so that both are consistent
            # 1. read nc file
            ds = OuceS5Data().read_ouce_s5_data(filepath)
        else:  # downloaded from CDSAPI as .grib
            # 1. read grib file
            ds = self.read_grib_file(filepath)

        # find all variables (sometimes download multiple)
        coords = [c for c in ds.coords]
        vars = [v for v in ds.variables if v not in coords]
        variable = '-'.join(vars)

        # 2. create the filepath and save to that location
        output_path = self.create_filename(
            filepath,
            self.interim,
            variable,
            subset_name=subset_str if subset_str is not None else None
        )
        assert output_path.name[-3:] == '.nc', f'\
        filepath name should be a .nc file. Currently: {filepath.name}'

        # IF THE FILE ALREADY EXISTS SKIP
        if output_path.exists():
            print(f"{output_path.name} already exists! Skipping.")
            return output_path, variable

        # 3. rename coords
        if 'latitude' in coords:
            ds = ds.rename({'latitude': 'lat'})
        if 'longitude' in coords:
            ds = ds.rename({'longitude': 'lon'})

        # 4. subset ROI
        if subset_str is not None:
            try:
                ds = self.chop_roi(ds, subset_str)
            except AssertionError:
                ds = self.chop_roi(ds, subset_str, inverse_lat=True)

        # 5. regrid (one variable at a time)
        if regrid is not None:
            assert all(np.isin(['lat', 'lon'], [c for c in ds.coords])), f"\
            Expecting `lat` `lon` to be in ds. dims : {[c for c in ds.coords]}"

            # regrid each variable individually
            all_vars = []
            for var in vars:
                if self.parallel:
                    # if parallel need to recreate new file each time
                    d_ = self.regrid(
                        ds[var].to_dataset(name=var), regrid,
                        clean=True, reuse_weights=False
                    )
                else:
                    d_ = self.regrid(
                        ds[var].to_dataset(name=var), regrid,
                        clean=False, reuse_weights=True,
                    )
                all_vars.append(d_)
            # merge the variables into one dataset
            ds = xr.merge(all_vars).sortby('initialisation_date')

        # 6. save ds to output_path
        ds.to_netcdf(output_path)
        return output_path, variable

    def merge_and_resample(self,
                           variable: str,
                           resample_str: Optional[str] = 'M',
                           upsampling: bool = False,
                           subset_str: Optional[str] = None) -> Path:
        # open all interim processed files (all variables?)
        ds = xr.open_mfdataset(self.interim.as_posix() + "/*.nc")
        ds = ds.sortby('initialisation_date')

        # resample
        if resample_str is not None:
            ds = self.resample_time(
                ds, resample_str, upsampling,
                time_coord='initialisation_date'
            )

        # save to preprocessed netcdf
        out_path = self.out_dir / f"{self.dataset}_{variable}_{subset_str}.nc"
        ds.to_netcdf(out_path)

        return out_path

    def preprocess(self, subset_str: Optional[str] = 'kenya',
                   regrid: Optional[Path] = None,
                   resample_time: Optional[str] = 'M',
                   upsampling: bool = False,
                   variable: Optional[str] = None,
                   cleanup: bool = False) -> None:
        """Preprocesses the S5 data for all variables in the 'ds' file at once

        Argument:
        ---------
        subset_str: Optional[str] = 'kenya'
            whether to subset the data to a particular region

        regrid: Optional[Path] = None
            whether to regrid to the same lat/lon grid as the `regrid` ds

        resample_time: Optional[str] = 'M'
            whether to resample the timesteps to a given `frequency`

        upsampling: bool = False
            are you upsampling the time frequency (e.g. monthly -> daily)

        variable: Optional[str] = None
            if self.ouce_server then require a variable string to build
            the filepath to the data to preprocess

        cleanup: bool = False
            Whether to cleanup the self.interim directory
        """
        if self.ouce_server:
            # data already in netcdf but needs other preprocessing
            assert variable is not None, f"Must pass a variable argument when\
            preprocessing the S5 data on the OUCE servers"
            os = OuceS5Data()
            filepaths = os.get_ouce_filepaths(variable=variable)
        else:
            filepaths = self.get_filepaths()

        if not self.parallel:
            out_paths = []
            variables = []
            for filepath in filepaths:
                output_path, variable = self._preprocess(
                    filepath, subset_str, regrid
                )
                out_paths.append(output_path)
                variables.append(variable)
        else:
            # Not implemented parallel yet
            pool = multiprocessing.Pool(processes=100)
            outputs = pool.map(
                partial(self._preprocess,
                        subset_str=subset_str,
                        regrid=regrid),
                filepaths)
            pass

        # merge all of the timesteps for S5 data
        for var in np.unique(variables):
            cast(str, var)
            self.merge_and_resample(
                var, resample_time, upsampling, subset_str
            )

        if cleanup:
            rmtree(self.interim)
