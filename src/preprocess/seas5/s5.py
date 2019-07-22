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

    def get_filepaths(self, target_folder: Path,  # type: ignore
                      variable: str,
                      grib: bool = True) -> List[Path]:
        # if target_folder.name == 'raw':
        #     target_folder = self.raw_folder

        pattern = f'seasonal*/{variable}/*/*.grib' if grib else f'*/{variable}/*/*.nc'
        # get all files in */*/*
        outfiles = [f for f in target_folder.glob(pattern)]

        outfiles.sort()
        return outfiles

    @staticmethod
    def read_grib_file(filepath: Path) -> xr.Dataset:
        assert filepath.suffix in ['.grib', '.grb'], f"This method is for \
        `grib` files. Not for {filepath.name}"
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
        output_path = output_dir / variable / filename
        if not output_path.parents[0].exists():
            (output_path.parents[0]).mkdir(exist_ok=True, parents=True)
        return output_path

    def _preprocess(self,
                    filepath: Path,
                    ouce_server: bool = False,
                    subset_str: Optional[str] = None,
                    regrid: Optional[xr.Dataset] = None,
                    **kwargs) -> Tuple[Path, str]:
        """preprocess a single s5 dataset (multi-variables per `.nc` file)"""
        print(f"\nWorking on {filepath.name}")

        if self.ouce_server:
            # undoes the preprocessing so that both are consistent
            infer = kwargs.pop('infer') if 'infer' in kwargs.keys() else False
            # 1. read nc file
            ds = OuceS5Data().read_ouce_s5_data(filepath, infer=infer)
        else:  # downloaded from CDSAPI as .grib
            # 1. read grib file
            ds = self.read_grib_file(filepath)
            # add initialisation_date as a dimension
            ds = ds.expand_dims(dim='initialisation_date')

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
        ds = xr.open_mfdataset((self.interim / variable).as_posix() + "/*.nc")
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

    def preprocess(self, variable: str,
                   regrid: Optional[Path] = None,
                   subset_str: Optional[str] = 'kenya',
                   resample_time: Optional[str] = 'M',
                   upsampling: bool = False,
                   cleanup: bool = False,
                   **kwargs) -> None:
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

        variable: str
            Which variable do we want to process? [NOTE: each call
            to `obj.preprocess()` preprocesses ONE variable]

            if self.ouce_server then also require a variable string
            to build the filepath to the data to preprocess

        cleanup: bool = False
            Whether to cleanup the self.interim directory

        kwargs: dict
            keyword arguments (mostly for pytest!)
            'ouce_dir' : Path - the test directory to use for reading .nc files
            'infer' : bool - whether to infer the frequency when creating
                the forecast horizon for ouce_data.
        """
        if self.ouce_server:
            # data already in netcdf but needs other preprocessing
            assert variable is not None, f"Must pass a variable argument when\
            preprocessing the S5 data on the OUCE servers"
            os = OuceS5Data()

            # get kwargs
            ouce_dir = kwargs.pop('ouce_dir') if 'ouce_dir' in kwargs.keys() else None

            filepaths = os.get_ouce_filepaths(
                variable=variable, parent_dir=ouce_dir
            )
        else:
            filepaths = self.get_filepaths(
                target_folder=self.raw_folder, variable=variable
            )

        # argument needs the regrid file
        if regrid is not None:
            regrid_ds = xr.open_dataset(regrid)
        else:
            regrid_ds = None

        if not self.parallel:
            out_paths = []
            variables = []
            for filepath in filepaths:
                output_path, variable = self._preprocess(
                    filepath, subset_str, regrid_ds, self.ouce_server, **kwargs
                )
                out_paths.append(output_path)
                variables.append(variable)

        else:
            # Not implemented parallel yet
            pool = multiprocessing.Pool(processes=100)
            outputs = pool.map(
                partial(self._preprocess,
                        subset_str=subset_str,
                        regrid=regrid,
                        ouce_server=self.ouce_server),
                filepaths)
            print("\nOutputs (errors):\n\t", outputs)

        # merge all of the timesteps for S5 data
        for var in np.unique(variables):
            cast(str, var)
            self.merge_and_resample(
                var, resample_time, upsampling, subset_str
            )

        if cleanup:
            rmtree(self.interim)
