import numpy as np
import xarray as xr

from src.preprocess import SRTMPreprocessor
from src.utils import get_kenya

from ..utils import _make_dataset


class TestSRTMPreprocessor:

    @staticmethod
    def _make_srtm_dataset(size, lonmin=-180.0, lonmax=180.0,
                           latmin=-55.152, latmax=75.024):
        lat_len, lon_len = size
        # create the vector
        longitudes = np.linspace(lonmin, lonmax, lon_len)
        latitudes = np.linspace(latmin, latmax, lat_len)

        dims = ['lat', 'lon']
        coords = {'lat': latitudes,
                  'lon': longitudes}

        crs = np.random.randint(10, size=size)
        Band1 = np.random.randint(10, size=size)

        # make datast with correct attrs
        ds = xr.Dataset(
            {
                'crs': (dims, crs),
                'Band1': (dims, Band1),
            },
            coords=coords
        )
        return ds

    def test_init(self, tmp_path):

        _ = SRTMPreprocessor(tmp_path)

        static_folder = tmp_path / 'interim/static'
        assert static_folder.exists(), 'Static folder not created'
        assert (static_folder / 'srtm_preprocessed').exists()
        assert (static_folder / 'srtm_interim').exists()

    def test_preprocess(self, tmp_path):

        (tmp_path / 'raw/srtm').mkdir(parents=True)
        data_path = tmp_path / 'raw/srtm/kenya.nc'
        dataset = self._make_srtm_dataset(size=(100, 100))
        dataset.to_netcdf(path=data_path)

        kenya = get_kenya()
        regrid_dataset, _, _ = _make_dataset(size=(20, 20),
                                             latmin=kenya.latmin, latmax=kenya.latmax,
                                             lonmin=kenya.lonmin, lonmax=kenya.lonmax)

        regrid_path = tmp_path / 'regridder.nc'
        regrid_dataset.to_netcdf(regrid_path)

        processor = SRTMPreprocessor(tmp_path)

        processor.preprocess(subset_str='kenya', regrid=regrid_path)

        expected_out_processed = (
            tmp_path / 'interim/static/srtm_preprocessed/kenya.nc'
        )
        assert expected_out_processed.exists(), \
            'expected a processed folder'

        # check the subsetting happened correctly
        out_data = xr.open_dataset(expected_out_processed)
        expected_dims = ['lat', 'lon']
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

        assert out_data.topography.values.shape == (20, 20)

        assert not processor.interim.exists(), \
            f'Interim srtm folder should have been deleted'
