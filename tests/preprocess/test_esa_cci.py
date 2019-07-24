import xarray as xr
import numpy as np
from datetime import datetime
import pytest

from src.preprocess import ESACCIPreprocessor
from src.utils import get_kenya, get_ethiopia

from ..utils import _make_dataset


class TestESACCIPreprocessor:

    @staticmethod
    def test_make_filename():
        test_file = '1982-testy_test.nc'
        expected_output = '1982_1982-testy_test_kenya.nc'

        filename = ESACCIPreprocessor().create_filename(test_file, 'kenya')
        assert filename == expected_output, \
            f'Expected output to be {expected_output}, got {filename}'

    @staticmethod
    def _make_ESA_CCI_dataset(size, lonmin=-180.0, lonmax=180.0,
                              latmin=-55.152, latmax=75.024,
                              add_times=False):
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
        lccs_class = np.random.randint(100, size=size)

        # make datast with correct attrs
        ds = xr.Dataset({'lccs_class': (dims, lccs_class)}, coords=coords)
        ds.attrs['time_coverage_start'] = '20190101'

        return ds

    @staticmethod
    def test_directories_created(tmp_path):
        processor = ESACCIPreprocessor(tmp_path)

        assert (
            tmp_path / processor.preprocessed_folder / 'esa_cci_landcover_preprocessed'
        ).exists(), \
            'Should have created a directory tmp_path/interim' \
            '/esa_cci_landcover_preprocessed'

        assert (
            tmp_path / processor.preprocessed_folder / 'esa_cci_landcover_interim'
        ).exists(), \
            'Should have created a directory tmp_path/interim' \
            '/esa_cci_landcover_interim'

    @staticmethod
    def test_get_filenames(tmp_path):

        (tmp_path / 'raw' / 'esa_cci_landcover').mkdir(parents=True)

        test_file = tmp_path / 'raw/esa_cci_landcover/1992-v2.0.7b_testy_test.nc'
        test_file.touch()

        processor = ESACCIPreprocessor(tmp_path)

        files = processor.get_filepaths()
        assert files[0] == test_file, f'Expected {test_file} to be retrieved'

    @pytest.mark.parametrize('cleanup', [True, False])
    def test_preprocess(self, tmp_path, cleanup):

        (tmp_path / 'raw/esa_cci_landcover').mkdir(parents=True)
        data_path = tmp_path / 'raw/esa_cci_landcover/1992-v2.0.7b_testy_test.nc'
        dataset = self._make_ESA_CCI_dataset(size=(100, 100))
        dataset.to_netcdf(path=data_path)

        kenya = get_kenya()
        regrid_dataset, _, _ = _make_dataset(size=(20, 20),
                                             latmin=kenya.latmin, latmax=kenya.latmax,
                                             lonmin=kenya.lonmin, lonmax=kenya.lonmax)

        regrid_path = tmp_path / 'regridder.nc'
        regrid_dataset.to_netcdf(regrid_path)

        processor = ESACCIPreprocessor(tmp_path)
        processor.preprocess(subset_str='kenya', regrid=regrid_path,
                             parallel_processes=1, cleanup=cleanup)

        expected_out_path = (
            tmp_path / 'interim/esa_cci_landcover_interim'
            '/1992_1992-v2.0.7b_testy_test_kenya.nc'
        )
        if not cleanup:
            assert expected_out_path.exists(), \
                f'Expected processed file to be saved to {expected_out_path}'

        expected_out_processed = (
            tmp_path / 'interim' / 'esa_cci_landcover_'
            'preprocessed' / 'esa_cci_landcover_kenya.nc'
        )
        assert expected_out_processed.exists(), \
            'expected a processed folder'

        # check the subsetting happened correctly
        out_data = xr.open_dataset(expected_out_processed)
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

        assert out_data.lc_class.values.shape[1:] == (20, 20)

        if cleanup:
            assert not processor.interim.exists(), \
                f'Interim esa_cci_landcover folder should have been deleted'

    def test_alternative_region(self, tmp_path):
        # make the dataset
        (tmp_path / 'raw/esa_cci_landcover').mkdir(parents=True)
        data_path = tmp_path / 'raw/esa_cci_landcover/1992-v2.0.7b_testy_test.nc'
        dataset = self._make_ESA_CCI_dataset(size=(100, 100))
        dataset.attrs['time_coverage_start'] = '20190101'
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
        processor = ESACCIPreprocessor(tmp_path)
        processor.preprocess(
            subset_str='ethiopia', regrid=regrid_path,
            parallel_processes=1
        )

        expected_out_path = (
            tmp_path / 'interim/esa_cci_landcover_preprocessed'
            '/esa_cci_landcover_ethiopia.nc'
        )
        assert expected_out_path.exists(), \
            f'Expected processed file to be saved to {expected_out_path}'
