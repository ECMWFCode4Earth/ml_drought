from pathlib import Path
from shutil import rmtree
import xarray as xr

from typing import Optional

from .base import BasePreProcessor


class SRTMPreprocessor(BasePreProcessor):
    dataset = 'srtm'
    static = True

    def preprocess(self, subset_str: str = 'kenya',
                   regrid: Optional[Path] = None,
                   cleanup: bool = True) -> None:
        """Preprocess a downloaded topography .nc file to produce
        one subset file with no timestep

        Arguments:
        ---------
        subset_str: str = 'kenya'
            Because the SRTM data can only be downloaded in tiles, the subsetting happens
            during the export step. This tells the preprocessor which file to preprocess
        regrid: Optional[Path] = None
            If a Path is passed, the CHIRPS files will be regridded to have the same
            grid as the dataset at that Path. If None, no regridding happens
        cleanup: bool = True
            If true, delete interim files created by the class

        """
        print(f'Reading data from {self.raw_folder}. Writing to {self.interim}')

        if regrid is not None:
            regrid = self.load_reference_grid(regrid)

        netcdf_filepath = self.raw_folder / f'{self.dataset}/{subset_str}.nc'

        print(f'Starting work on {netcdf_filepath.name}')
        ds = xr.open_dataset(netcdf_filepath).drop('crs').rename({'Band1': 'topography'})

        if regrid is not None:
            # using the default 'nearest_s2d' method doesn't work
            # The ESMF regrid function returns lots of internal
            # subroutine errors
            ds = self.regrid(ds, regrid, method='bilinear')

        print(f'Saving to {self.out_dir}/{subset_str}.nc')
        ds.to_netcdf(str(self.out_dir / f'{subset_str}.nc'))

        print(f'Processed {netcdf_filepath}')

        if cleanup:
            rmtree(self.interim)
