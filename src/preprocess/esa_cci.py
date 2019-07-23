from pathlib import Path
import xarray as xr

from .base import BasePreProcessor


class ESACCIPreprocessor(BasePreProcessor):
    """ Preprocesses the ESA CCI Landcover data """
    dataset = 'esa_cci_landcover'

    def create_filename():
        pass

    def merge_and_resample():
        pass

    def preprocess(self, subset_str: Optional[str] = 'kenya',
                   regrid: Optional[Path] = None,
                   resample_time: Optional[str] = 'M',
                   upsampling: bool = False,
                   parallel: bool = False,
                   cleanup: bool = True) -> None:
        """Preprocess all of the ESA CCI landcover .nc files to produce
        one subset file resampled to the timestep of interest.
        (downloaded as annual timesteps)

        Note:
        ----
        - because the landcover data only goes back to 1993 for all dates
        before 1993 that we need data for  we have selected the `modal`
        class from the whole data range (1993-2019).
        - This assumes that landcover is relatively consistent in the 1980s
        as the 1990s, 2000s and 2010s
        """
        print(f'Reading data from {self.raw_folder}. Writing to {self.interim}')

        nc_files = self.get_filepaths()
        pass
