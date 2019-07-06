from pathlib import Path
from functools import partial
import xarray as xr
import multiprocessing
from shutil import rmtree
from typing import Optional, List

from ..base import BasePreProcessor
from .fcast_horizon import FH
from .ouce_s5 import OuceS5Data

class S5Preprocessor(BasePreProcessor):

    dataset = 's5'

    def __init__(self, data_folder: Path = Path('data'),
                 ouce_server: bool = False) -> None:
        super().__init__(data_folder)
        self.ouce_server = ouce_server
    #
    def get_filepaths(self, folder: str = 'raw') -> List[Path]:
        """ because reading .grib files have to rewrite get_filepaths"""
        if folder == 'raw':
            target_folder = self.raw_folder / self.dataset
        else:
            target_folder = self.interim
        outfiles = list(target_folder.glob('**/*.grib'))
        outfiles.sort()
        return outfiles

    def read_grib_file(filepath: Path) -> xr.Dataset:
        assert filepath.suffix = '.grib', f"This method is for `.grib` files\
        Not for {filepath.as_posix()}"
        ds = xr.open_dataset(filepath, engine='cfgrib')
        return ds

    def create_filename(netcdf_filepath: Path,
                        output_dir: Path,
                        variable: str,
                        filepath: Path) -> Path:
        # TODO: do we want each variable in separate folders / .nc files?
        filename = netcdf_filepath.stem + variable + '.nc'
        output_path = output_dir / variable / filename
        return output_path

    def _preprocess_one_var(ds: xr.Dataset,
                            variable: str,) -> Path:
        # 2. subset ROI
        if subset_str is not None:
            ds = self.chop_roi(ds, subset_str)

        # 3. regrid
        if regrid is not None:
            ds = self.regrid(ds, regrid)

        # 4. create the filepath and save to that location
        output_path = self.create_filename(
            netcdf_filepath,
            self.output_dir,
            variable,
            subset_name=subset_str if subset_str is not None else None
        )
        assert output_path.name[-3:] == '.nc', \
        f'filepath name should be a .nc file. Currently: {netcdf_filepath.name}'

        # 5. save ds to output_path
        ds.to_netcdf(output_path)

        return output_path

    def _preprocess(filepath: Path,
                    ouce_server: bool = False) -> List[Path]:
        """preprocess a single s5 dataset (may have many variables)"""
        if ouce_server:
            # undoes the preprocessing so that both are consistent
            # 1. read nc file
            ds = OuceS5Data.read_ouce_s5_data(filepath)
        else: # downloaded from CDSAPI as .grib
            # 1. read grib file
            ds = read_grib_file(filepath)

        # find all variables (sometimes download multiple)
        coords = [c for c in ds.coords]
        vars = [v for v in ds.variables if v not in coords]

        output_paths = []
        for var in vars:
            ds_one_var = ds[var].to_dataset(name=var)
            output_paths.append(_preprocess_one_var(ds_one_var, var, filepath))

        return output_paths

    def merge_and_resample(self,
                           variable: str,
                           resample_length: Optional[str] = 'M',
                           upsampling: bool = False,) -> Path:
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
                   parallel: bool = False,) -> None:
        if self.ouce_server:
            # data already in netcdf but needs other preprocessing
            os = OuceS5Data()
            filepaths = os.get_ouce_filepaths(variable=???)
        else:
            filepaths = get_filepaths()

        self._preprocess(filepath, self.ouce_server)
        self.merge_and_resample()

        pass
