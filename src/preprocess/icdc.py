from pathlib import Path
import xarray as xr
from shutil import rmtree
from typing import Optional

from .base import BasePreProcessor


class ICDCPreprocessor(BasePreProcessor):
    """ For working with data on ICDC (TEMP.)
    """
    variable: str
    icdc_data_dir = Path('/pool/data/ICDC/land/')

    def get_filepaths(self) -> List[Path]:
        dir = self.icdc_data_dir / self.variable / 'DATA'
        years = [d.name for d in dir.iterdir() if d.is_dir()]

        filepaths = []
        for year in years:
            filepaths.extend((dir / year).glob('*.nc'))

        return filepaths

    @staticmethod
    def create_filename(netcdf_filename: str,
                        subset_name: Optional[str] = None) -> str:
        """
        {base_str}.nc
        """
        filename_stem = netcdf_filename[:-3]
        if subset_name is not None:
            new_filename = f'{filename_stem}_{subset_name}.nc'
        else:
            new_filename = f'{filename_stem}.nc'
        return new_filename

    def _preprocess_single(self, netcdf_filepath: Path,
                           subset_str: Optional[str] = 'kenya',
                           regrid: Optional[xr.Dataset] = None) -> None:
        """Run the Preprocessing steps for the GLEAM data

        Process:
        -------
        * chop out ROI
        * create new dataset with regrid dimensions
        * Save the output file to new folder
        """
        print(f'Starting work on {netcdf_filepath.name}')
        # 1. read in the dataset
        ds = xr.open_dataset(netcdf_filepath)

        # 2. chop out EastAfrica
        if subset_str is not None:
            ds = self.chop_roi(ds, subset_str, inverse_lat=True)

        if regrid is not None:
            ds = self.regrid(ds, regrid)

        # 6. create the filepath and save to that location
        assert netcdf_filepath.name[-3:] == '.nc', \
            f'filepath name should be a .nc file. Currently: {netcdf_filepath.name}'

        filename = self.create_filename(
            netcdf_filepath.name,
            subset_name=subset_str if subset_str is not None else None
        )
        print(f"Saving to {self.interim}/{filename}")
        ds.to_netcdf(self.interim / filename)

        print(f"** Done for {self.variable} {netcdf_filepath.name} **")

    def merge_files(self):
        pass

    def preprocess(self, subset_str: Optional[str] = 'kenya',
                   regrid: Optional[Path] = None,
                   resample_time: Optional[str] = 'M',
                   upsampling: bool = False,
                   cleanup: bool = True) -> None:
        """ Preprocess all of the GLEAM .nc files to produce
        one subset file.

        Arguments
        ----------
        subset_str: Optional[str] = 'kenya'
            Whether to subset Kenya when preprocessing
        regrid: Optional[Path] = None
            If a Path is passed, the CHIRPS files will be regridded to have the same
            grid as the dataset at that Path. If None, no regridding happens
        resample_time: str = 'M'
            If not None, defines the time length to which the data will be resampled
        upsampling: bool = False
            If true, tells the class the time-sampling will be upsampling. In this case,
            nearest instead of mean is used for the resampling
        cleanup: bool = True
            If true, delete interim files created by the class
        """
        nc_files = self.get_filepaths()

        if regrid is not None:
            regrid = self.load_reference_grid(regrid)

        for file in nc_files:
            self._preprocess_single(file, subset_str, regrid)

        # merge all of the timesteps
        self.merge_files(subset_str, resample_time, upsampling)

        if cleanup:
            rmtree(self.interim)


class ESACCISoilMoisture(ICDCPreprocessor):
    variable = 'esa_cci_soilmoisture'


class LAIModisAvhrr(ICDCPreprocessor):
    variable = 'avhrr_modis_lai'


class ModisNDVI(ICDCPreprocessor):
    variable = 'modis_aqua_vegetationindex'
