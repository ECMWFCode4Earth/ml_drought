import xarray as xr
import numpy as np

from src.preprocess import CHIRPSPreprocesser
from src.utils import get_kenya

from .test_utils import _make_dataset


class TestCHIRPSPreprocessor:

    @staticmethod
    def _make_chirps_dataset(size, lonmin=-180.0, lonmax=180.0,
                             latmin=-55.152, latmax=75.024,
                             add_times=True):
        lat_len, lon_len = size
        # create the vector
        longitudes = np.linspace(lonmin, lonmax, lon_len)
        latitudes = np.linspace(latmin, latmax, lat_len)

        dims = ['latitude', 'longitude']
        coords = {'latitude': latitudes,
                  'longitude': longitudes}

        if add_times:
            size = (2, size[0], size[1])
            dims.insert(0, 'time')
            coords['time'] = [0, 1]
        vhi = np.random.randint(100, size=size)

        return xr.Dataset({'VHI': (dims, vhi)}, coords=coords)

    @staticmethod
    def test_directories_created(tmp_path):
        v = CHIRPSPreprocesser(tmp_path)

        assert (tmp_path / v.interim_folder / 'chirps_preprocessed').exists(), \
            'Should have created a directory tmp_path/interim/chirps_preprocessed'

        assert (tmp_path / v.interim_folder / 'chirps').exists(), \
            'Should have created a directory tmp_path/interim/chirps'

    @staticmethod
    def test_get_filenames(tmp_path):

        (tmp_path / 'raw' / 'chirps').mkdir(parents=True)

        test_file = tmp_path / 'raw/chirps/testy_test.nc'
        test_file.touch()

        processor = CHIRPSPreprocesser(tmp_path)

        files = processor.get_chirps_filepaths()
        assert files[0] == test_file, f'Expected {test_file} to be retrieved'

    @staticmethod
    def test_make_filename():

        test_file = 'testy_test.nc'
        expected_output = 'testy_test_kenya.nc'

        filename = CHIRPSPreprocesser.create_filename(test_file, 'kenya')
        assert filename == expected_output, \
            f'Expected output to be {expected_output}, got {filename}'

    def test_preprocess(self, tmp_path):

        (tmp_path / 'raw/chirps/global').mkdir(parents=True)
        data_path = tmp_path / 'raw/chirps/global/testy_test.nc'
        dataset = self._make_chirps_dataset(size=(100, 100))
        dataset.to_netcdf(path=data_path)

        kenya = get_kenya()
        regrid_dataset, _, _ = _make_dataset(size=(1000, 1000),
                                             latmin=kenya.latmin, latmax=kenya.latmax,
                                             lonmin=kenya.lonmin, lonmax=kenya.lonmax)

        processor = CHIRPSPreprocesser(tmp_path)
        processor.preprocess(subset_kenya=True, regrid=regrid_dataset,
                             parallel=False)

        expected_out_path = tmp_path / 'interim/chirps/testy_test_kenya.nc'
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

        assert out_data.VHI.values.shape[1:] == (1000, 1000)
