from pathlib import Path
import xarray as xr
from shutil import rmtree
from typing import Optional, List

from .base import BasePreProcessor


class ICDCPreprocessor(BasePreProcessor):
    """ For working with data on ICDC (SPECIFIC to Uni Server)
    """
    variable: str  # the name of the variable on icdc
    source: str  # {'land', 'atmosphere', 'climate_indices', 'ocean', 'ice_and_snow'}

    def __init__(self, data_folder: Path = Path('data')) -> None:
        super().__init__(data_folder)
        self.icdc_data_dir = Path(f'/pool/data/ICDC/{self.source}/')

    def get_icdc_filepaths(self) -> List[Path]:
        dir = self.icdc_data_dir / self.dataset / 'DATA'
        years = [d.name for d in dir.iterdir() if d.is_dir()]

        filepaths: List = []
        for year in years:
            filepaths.extend((dir / year).glob('*.nc'))

        if filepaths != []:
            return filepaths
        else:
            filepaths.extend((dir).glob('*.nc'))
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
            try:
                ds = self.chop_roi(ds, subset_str, inverse_lat=True)
            except AssertionError:
                ds = self.chop_roi(ds, subset_str, inverse_lat=False)

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

        print(f"** Done for {self.dataset} {netcdf_filepath.name} **")

    def preprocess(self, subset_str: Optional[str] = 'kenya',
                   regrid: Optional[Path] = None,
                   resample_time: Optional[str] = 'M',
                   upsampling: bool = False,
                   cleanup: bool = False) -> None:
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
        nc_files = self.get_icdc_filepaths()

        if regrid is not None:
            regrid = self.load_reference_grid(regrid)

        for file in nc_files:
            self._preprocess_single(file, subset_str, regrid)

        # merge all of the timesteps
        self.merge_files(subset_str, resample_time, upsampling)

        if cleanup:
            rmtree(self.interim)


class ESACCISoilMoisturePreprocessor(ICDCPreprocessor):
    source = 'land'
    dataset = 'esa_cci_soilmoisture'


class LAIModisAvhrrPreprocessor(ICDCPreprocessor):
    source = 'land'
    dataset = 'avhrr_modis_lai'


class ModisNDVIPreprocessor(ICDCPreprocessor):
    source = 'land'
    dataset = 'modis_aqua_vegetationindex'


class AMSRESoilMoisturePreprocessor(ICDCPreprocessor):
    source = 'land'
    dataset = 'amsre_soilmoisture'


class ASCATSoilMoisturePreprocessor(ICDCPreprocessor):
    source = 'land'
    dataset = 'ascat_soilmoisture'


class EUMetsatAlbedoPreprocessor(ICDCPreprocessor):
    source = 'land'
    dataset = 'eumetsat_albedo'


class EUMetSatAlbedo2Preprocessor(ICDCPreprocessor):
    source = 'land'
    dataset = 'eumetsat_clara2_surfacealbedo'


class EUMetSatRadiationPreprocessor(ICDCPreprocessor):
    source = 'land'
    dataset = 'eumetsat_clara2_surfaceradiation'


class EUMetSatIrradiancePreprocessor(ICDCPreprocessor):
    source = 'land'
    dataset = 'eumetsat_surfacesolarirradiance'


class SpotFAPARPreprocessor(ICDCPreprocessor):
    source = 'land'
    dataset = 'fapar_spot_proba_v'


class GLEAMEvaporationPreprocessor(ICDCPreprocessor):
    source = 'land'
    dataset = 'gleam_evaporation'


class SpotLaiPreprocessor(ICDCPreprocessor):
    source = 'land'
    dataset = 'lai_spot_proba_v'


class SpotLSAlbedoPreprocessor(ICDCPreprocessor):
    source = 'land'
    dataset = 'land_surface_albedo_spot'


class ModisAlbedoPreprocessor(ICDCPreprocessor):
    source = 'land'
    dataset = 'modis_albedo'


class ModisForestCoverPreprocessor(ICDCPreprocessor):
    source = 'land'
    dataset = 'modis_forestcoverfraction'


class ModisLandcoverPreprocessor(ICDCPreprocessor):
    source = 'land'
    dataset = 'modis_landcover'


class ModisLatLonPreprocessor(ICDCPreprocessor):
    source = 'land'
    dataset = 'modis_latlon'


class ModisLSTClimatologyPreprocessor(ICDCPreprocessor):
    source = 'land'
    dataset = 'modis_lst_climatology'


class ModisNPPPreprocessor(ICDCPreprocessor):
    source = 'land'
    dataset = 'modis_primary_production'


class ModisSRTMPreprocessor(ICDCPreprocessor):
    source = 'land'
    dataset = 'modis-srtm_landwaterdistribution'


class ModisLSTPreprocessor(ICDCPreprocessor):
    source = 'land'
    dataset = 'modis_terra_landsurfacetemperature'


class SMOSSoilMoisturePreprocessor(ICDCPreprocessor):
    source = 'land'
    dataset = 'smos_soilmoisture'


class TopographyPreprocessor(ICDCPreprocessor):
    source = 'land'
    dataset = 'topography'


class SpotVegetationCoverFractionPreprocessor(ICDCPreprocessor):
    source = 'land'
    dataset = 'vegetationcoverfraction_spot_proba_v'
