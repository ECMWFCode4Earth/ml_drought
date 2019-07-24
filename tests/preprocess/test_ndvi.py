import xarray as xr
import numpy as np
from datetime import datetime

from src.preprocess import NDVIPreprocessor
from src.utils import get_kenya, get_ethiopia

from ..utils import _make_dataset


class TestNDVIPreprocessor:

    @staticmethod
    def test_make_filename():
        test_file = '19860702_testy-test.nc'
        expected_output = '1986_19860702_testy-test_kenya.nc'

        filename = NDVIPreprocessor().create_filename(test_file, 'kenya')
        assert filename == expected_output, \
            f'Expected output to be {expected_output}, got {filename}'

    @staticmethod
    def _make_ndvi_dataset(size, lonmin=-180.0, lonmax=180.0,
                           latmin=-55.152, latmax=75.024,
                           add_times=True):
        lat_len, lon_len = size
        # create the vector
        longitudes = np.linspace(lonmin, lonmax, lon_len)
        latitudes = np.linspace(latmin, latmax, lat_len)

        dims = ['lat', 'lon']
        coords = {'lat': latitudes,
                  'lon': longitudes,
                  'ncrs': [1],
                  'nv': [1]}

        if add_times:
            size = (1, size[0], size[1])
            dims.insert(0, 'time')
            coords['time'] = [datetime(2019, 1, 1)]
        ndvi = np.random.randint(100, size=size)

        return xr.Dataset({'NDVI': (dims, ndvi)}, coords=coords)

    @staticmethod
    def test_directories_created(tmp_path):
        processor = NDVIPreprocessor(tmp_path)

        assert (
            tmp_path / processor.preprocessed_folder / 'ndvi_preprocessed'
        ).exists(), \
            'Should have created a directory tmp_path/interim' \
            '/ndvi_preprocessed'

        assert (
            tmp_path / processor.preprocessed_folder / 'ndvi_interim'
        ).exists(), \
            'Should have created a directory tmp_path/interim' \
            '/ndvi_interim'

    @staticmethod
    def test_get_filenames(tmp_path):

        (tmp_path / 'raw' / 'ndvi').mkdir(parents=True)

        test_file = tmp_path / 'raw/ndvi/testy_test.nc'
        test_file.touch()

        processor = NDVIPreprocessor(tmp_path)

        files = processor.get_filepaths()
        assert files[0] == test_file, f'Expected {test_file} to be retrieved'

    def test_preprocess(self, tmp_path):

        (tmp_path / 'raw/ndvi/global').mkdir(parents=True)
        data_path = tmp_path / 'raw/ndvi/global/testy_test.nc'
        dataset = self._make_ndvi_dataset(size=(100, 100))
        dataset.to_netcdf(path=data_path)

        kenya = get_kenya()
        regrid_dataset, _, _ = _make_dataset(size=(20, 20),
                                             latmin=kenya.latmin, latmax=kenya.latmax,
                                             lonmin=kenya.lonmin, lonmax=kenya.lonmax)

        regrid_path = tmp_path / 'regridder.nc'
        regrid_dataset.to_netcdf(regrid_path)

        processor = NDVIPreprocessor(tmp_path)
        processor.preprocess(subset_str='kenya', regrid=regrid_path,
                             parallel_processes=1)

        expected_out_path = (
            tmp_path / 'interim/ndvi_preprocessed'
            '/ndvi_kenya.nc'
        )
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

        assert out_data.ndvi.values.shape[1:] == (20, 20)

        assert not processor.interim.exists(), \
            f'Interim ndvi folder should have been deleted'

    def test_alternative_region(self, tmp_path):
        # make the dataset
        (tmp_path / 'raw/ndvi').mkdir(parents=True)
        data_path = tmp_path / 'raw/ndvi/testy_test.nc'
        dataset = self._make_ndvi_dataset(size=(100, 100))
        dataset.to_netcdf(path=data_path)
        ethiopia = get_ethiopia()

        # regrid the datasets
        regrid_dataset, _, _ = _make_dataset(
            size=(20, 20), latmin=ethiopia.latmin,
            latmax=ethiopia.latmax, lonmin=ethiopia.lonmin,
            lonmax=ethiopia.lonmax
        )
        regrid_path = tmp_path / 'regridder.nc'
        regrid_dataset.to_netcdf(regrid_path)

        # build the Preprocessor object and subset with a different subset_str
        processor = NDVIPreprocessor(tmp_path)
        processor.preprocess(
            subset_str='ethiopia', regrid=regrid_path,
            parallel_processes=1
        )

        expected_out_path = (
            tmp_path / 'interim/ndvi_preprocessed'
            '/ndvi_ethiopia.nc'
        )
        assert expected_out_path.exists(), \
            f'Expected processed file to be saved to {expected_out_path}'
