import xarray as xr
import numpy as np
from datetime import datetime

from src.preprocess.icdc import ICDCPreprocessor, LAIModisAvhrrPreprocessor
from src.utils import get_kenya
from ..utils import _make_dataset


class TestICDCPreprocessor:

    @staticmethod
    def test_make_filename():
        test_file = 'testy_test.nc'
        expected_output = 'testy_test_kenya.nc'

        filename = ICDCPreprocessor.create_filename(test_file, 'kenya')
        assert filename == expected_output, \
            f'Expected output to be {expected_output}, got {filename}'

    @staticmethod
    def _make_icdc_dataset(size, lonmin=-180.0, lonmax=180.0,
                           latmin=-55.152, latmax=75.024,
                           add_times=True):
        lat_len, lon_len = size
        # create the vector
        longitudes = np.linspace(lonmin, lonmax, lon_len)
        latitudes = np.linspace(latmin, latmax, lat_len)

        dims = ['lat', 'lon']
        coords = {'lat': latitudes,
                  'lon': longitudes}

        if add_times:
            size = (2, size[0], size[1])
            dims.insert(0, 'time')
            coords['time'] = [datetime(2019, 1, 1), datetime(2019, 1, 2)]
        values = np.random.randint(100, size=size)

        return xr.Dataset({'lai': (dims, values)}, coords=coords)

    def _save_icdc_data(self, fpath, size):
        ds = self._make_icdc_dataset(size)
        if not fpath.parents[0].exists():
            fpath.parents[0].mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(fpath)

    @staticmethod
    def test_directories_created(tmp_path):
        v = LAIModisAvhrrPreprocessor(tmp_path)

        assert (
            tmp_path / v.preprocessed_folder / 'avhrr_modis_lai_preprocessed'
        ).exists(), \
            'Should have created a directory tmp_path/interim/avhrr_modis_lai_preprocessed'

        assert (
            tmp_path / v.preprocessed_folder / 'avhrr_modis_lai_interim'
        ).exists(), \
            'Should have created a directory tmp_path/interim/chirps_interim'

    @staticmethod
    def test_get_filenames(tmp_path):
        icdc_data_dir = tmp_path / 'pool' / 'data' / 'ICDC'
        icdc_path = icdc_data_dir / 'avhrr_modis_lai' / 'DATA'
        (icdc_path).mkdir(parents=True)

        test_file = icdc_path / 'testy_test.nc'
        test_file.touch()

        processor = LAIModisAvhrrPreprocessor(tmp_path)

        # overwrite internal icdc_data_dir to mock behaviour
        processor.icdc_data_dir = icdc_data_dir

        files = processor.get_icdc_filepaths()
        assert files[0] == test_file, f'Expected {test_file} to be retrieved'

    def test_preprocess(self, tmp_path):
        icdc_data_dir = tmp_path / 'pool' / 'data' / 'ICDC'
        icdc_path = icdc_data_dir / 'avhrr_modis_lai' / 'DATA'
        (icdc_path).mkdir(parents=True)
        icdc_path = icdc_path / 'GlobMap_V01_LAI__2005097__UHAM-ICDC.nc'
        self._save_icdc_data(icdc_path, (50, 50))

        kenya = get_kenya()
        regrid_dataset, _, _ = _make_dataset(size=(20, 20),
                                             latmin=kenya.latmin, latmax=kenya.latmax,
                                             lonmin=kenya.lonmin, lonmax=kenya.lonmax)

        regrid_path = tmp_path / 'regridder.nc'
        regrid_dataset.to_netcdf(regrid_path)

        processor = LAIModisAvhrrPreprocessor(tmp_path)
        # overwrite internal icdc_data_dir to mock behaviour
        processor.icdc_data_dir = icdc_data_dir

        processor.preprocess(
            subset_str='kenya', regrid=regrid_path, cleanup=True
        )

        expected_out_path = tmp_path / 'interim/' \
            'avhrr_modis_lai_preprocessed/avhrr_modis_lai_kenya.nc'
        assert expected_out_path.exists(), \
            f'Expected processed file to be saved to {expected_out_path}'

        # check the subsetting happened correctly
        out_data = xr.open_dataset(expected_out_path)
        expected_dims = ['lat', 'lon', 'time']
        assert len(list(out_data.dims)) == len(expected_dims)
        for dim in expected_dims:
            assert dim in list(out_data.dims), \
                f'Expected {dim} to be in the processed dataset dims'

        lons = out_data.lon.values
        assert (lons.min() >= kenya.lonmin) and (lons.max() <= kenya.lonmax), \
            'Longitudes not correctly subset'

        lats = out_data.lat.values
        assert (lats.min() >= kenya.latmin) and (lats.max() <= kenya.latmax), \
            'Latitudes not correctly subset'

        assert out_data.lai.values.shape[1:] == (20, 20)

        assert not processor.interim.exists(), \
            f'Interim chirps folder should have been deleted'
