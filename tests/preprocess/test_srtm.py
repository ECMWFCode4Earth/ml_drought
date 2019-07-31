import numpy as np
import xarray as xr

from src.preprocess import SRTMPreprocessor


class TestSRTMPreprocessor:

    @staticmethod
    def _make_ESA_CCI_dataset(size, lonmin=-180.0, lonmax=180.0,
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
