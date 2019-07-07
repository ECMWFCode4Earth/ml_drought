from pathlib import Path
import xarray as xr
# from functools import partial
# import multiprocessing
# from shutil import rmtree
from typing import Optional, List, Tuple

from ..base import BasePreProcessor
from .ouce_s5 import OuceS5Data


class S5Preprocessor(BasePreProcessor):
    dataset: str = 's5'

    def __init__(self, data_folder: Path = Path('data'),
                 ouce_server: bool = False) -> None:
        super().__init__(data_folder)
        self.ouce_server = ouce_server

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
        assert filepath.suffix == '.grib', f"This method is for `.grib` files\
        Not for {filepath.as_posix()}"
        ds = xr.open_dataset(filepath, engine='cfgrib')
        return ds

    @staticmethod
    def create_filename(filepath: Path,
                        output_dir: Path,
                        variable: str,
                        regrid: Optional[xr.Dataset] = None,
                        subset_name: Optional[str] = None) -> Path:
        # TODO: do we want each variable in separate folders / .nc files?
        subset_name = ('_' + subset_name) if subset_name is not None else ''
        filename = filepath.stem + f'{variable}{subset_name}.nc'
        output_path = output_dir / filename
        return output_path

    def _preprocess_one_var(self, ds: xr.Dataset,
                            variable: str,
                            filepath: Path,
                            subset_str: Optional[str],
                            regrid: Optional[bool] = None) -> Path:
        # 2. subset ROI
        if subset_str is not None:
            ds = self.chop_roi(ds, subset_str)

        # 3. regrid
        if regrid is not None:
            ds = self.regrid(ds, regrid)

        # 4. create the filepath and save to that location
        output_path = self.create_filename(
            filepath=filepath,
            output_dir=self.out_dir,
            variable=variable,
            subset_name=subset_str if subset_str is not None else None
        )
        assert output_path.name[-3:] == '.nc', f'\
        filepath name should be a .nc file. Currently: {filepath.name}'

        # 5. save ds to output_path
        ds.to_netcdf(output_path)

        return output_path

    def _preprocess_vars_separately(self,
                                    filepath: Path,
                                    ouce_server: bool = False,
                                    regrid: Optional[xr.Dataset] = None,
                                    subset_str: Optional[str] = None,
                                    ) -> List[Path]:
        """preprocess a single s5 dataset (each variable separately)"""
        if ouce_server:
            # undoes the preprocessing so that both are consistent
            # 1. read nc file
            o = OuceS5Data()
            ds = o.read_ouce_s5_data(filepath)
        else:  # downloaded from CDSAPI as .grib
            # 1. read grib file
            ds = self.read_grib_file(filepath)

        # find all variables (sometimes download multiple)
        coords = [c for c in ds.coords]
        vars = [v for v in ds.variables if v not in coords]

        output_paths = []
        for var in vars:
            ds_one_var = ds[var].to_dataset(name=var)
            output_paths.append(self._preprocess_one_var(
                ds=ds_one_var, variable=var, filepath=filepath,
                subset_str=subset_str, regrid=regrid
            ))

        return output_paths

    def _preprocess(self,
                    filepath: Path,
                    subset_str: Optional[str] = None,
                    regrid: Optional[xr.Dataset] = None) -> Tuple[Path, str]:
        """preprocess a single s5 dataset (multi-variables per `.nc` file)"""
        print(f"working on {filepath.name}")

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

        if 'latitude' in coords:
            ds = ds.rename({'latitude': 'lat'})
        if 'longitude' in coords:
            ds = ds.rename({'longitude': 'lon'})

        # 2. subset ROI
        if subset_str is not None:
            ds = self.chop_roi(ds, subset_str)

        # 3. regrid
        if regrid is not None:
            assert all(np.isin(['lat', 'lon'], [c for c in ds.coords])), f"\
            Expecting `lat` `lon` to be in ds. dims : {[c for c in ds.coords]}"
            ds = self.regrid(ds, regrid)

        # 4. create the filepath and save to that location
        output_path = self.create_filename(
            filepath,
            self.out_dir,
            variable,
            subset_name=subset_str if subset_str is not None else None
        )
        assert output_path.name[-3:] == '.nc', f'\
        filepath name should be a .nc file. Currently: {filepath.name}'

        # 5. save ds to output_path
        ds.to_netcdf(output_path)
        return output_path, variable

    def merge_and_resample(self,
                           variable: str,
                           resample_length: Optional[str] = 'M',
                           upsampling: bool = False,
                           subset_str: Optional[str] = None) -> Path:
        # open all interim processed files (all variables?)
        ds = xr.open_mfdataset(self.interim / "*.nc")

        # resample
        if resample_length is not None:
            ds.resample_time(resample_length, upsampling)

        # save to preprocessed netcdf
        out_path = self.out_dir / f"{self.dataset}_{variable}_{subset_str}.nc"
        ds.to_netcdf(out_path)

        return out_path

    def preprocess(self, subset_str: Optional[str] = 'kenya',
                   regrid: Optional[Path] = None,
                   resample_time: Optional[str] = 'M',
                   upsampling: bool = False,
                   parallel: bool = False,
                   variable: Optional[str] = None) -> None:
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

        parallel: bool = False
            whether to run in parallel

        variable: Optional[str] = None
            if self.ouce_server then require a variable string to build
            the filepath to the data to preprocess

        """
        if self.ouce_server:
            # data already in netcdf but needs other preprocessing
            assert variable is not None, f"Must pass a variable argument when\
            preprocessing the S5 data on the OUCE servers"
            os = OuceS5Data()
            filepaths = os.get_ouce_filepaths(variable=variable)
        else:
            filepaths = self.get_filepaths()

        if not parallel:
            out_paths = []
            variables = []
            for filepath in filepaths:
                output_path, variable = self._preprocess(
                    filepath, self.ouce_server, subset_str, regrid
                )
                out_paths.append(output_path)
                variables.append(variable)
        else:
            # Not implemented parallel yet
            pass

        # merge all of the timesteps for S5 data
        for variable in variables:
            self.merge_and_resample(
                variable, resample_time, upsampling, subset_str
            )
