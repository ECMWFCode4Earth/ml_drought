from src.models.base import ModelBase

from ..utils import _make_dataset


class TestBase:

    def test_init(self, tmp_path):
        ModelBase(tmp_path)
        assert (tmp_path / 'models').exists(), f'Models dir not made!'

    def test_ds_to_np(self, tmp_path):

        x, _, _ = _make_dataset(size=(5, 5))
        y, _, _ = _make_dataset(size=(5, 5))
        y = y.isel(time=[0])

        x.to_netcdf(tmp_path / 'x.nc')
        y.to_netcdf(tmp_path / 'y.nc')

        x_np, y_np, latlons = ModelBase.ds_folder_to_np(tmp_path, return_latlons=True)

        assert x_np.shape[0] == y_np.shape[0] == latlons.shape[0], \
            f'x, y and latlon data have a different number of instances! ' \
            f'x: {x_np.shape[0]}, y: {y_np.shape[0]}, latlons: {latlons.shape[0]}'

        for idx in range(latlons.shape[0]):

            lat, lon = latlons[idx, 0], latlons[idx, 1]

            for time in range(x_np.shape[1]):
                target = x.isel(time=time).sel(lat=lat).sel(lon=lon).VHI.values

                assert target == x_np[idx, time, 0], \
                    f'Got different x values for time idx: {time}, lat: {lat}, ' \
                    f'lon: {lon}.Expected {target}, got {x_np[idx, time, 0]}'

            target_y = y.isel(time=0).sel(lat=lat).sel(lon=lon).VHI.values
            assert target_y == y_np[idx, 0], \
                f'Got y different values for lat: {lat}, ' \
                f'lon: {lon}.Expected {target_y}, got {y_np[idx, 0]}'

    def test_load_train(self, tmp_path):

        assert 1 == 1
