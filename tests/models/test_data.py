import pytest

from src.models.data import _BaseIter

from ..utils import _make_dataset


class TestBaseIter:

    @pytest.mark.parametrize('normalize', [True, False])
    def test_ds_to_np(self, tmp_path, normalize):

        x, _, _ = _make_dataset(size=(5, 5))
        y = x.isel(time=[0])

        x.to_netcdf(tmp_path / 'x.nc')
        y.to_netcdf(tmp_path / 'y.nc')

        norm_dict = {}
        for var in x.data_vars:
            norm_dict[var] = {
                'mean': x[var].mean(dim=['lat', 'lon'], skipna=True).values,
                'std': x[var].std(dim=['lat', 'lon'], skipna=True).values
            }

        class MockLoader:
            def __init__(self):
                self.batch_file_size = None
                self.mode = None
                self.shuffle = None
                self.clear_nans = None
                self.data_files = []
                self.normalizing_dict = norm_dict if normalize else None

        base_iterator = _BaseIter(MockLoader())

        arrays = base_iterator.ds_folder_to_np(tmp_path, return_latlons=True)

        x_np, y_np, latlons = arrays.x, arrays.y, arrays.latlons

        assert x_np.shape[0] == y_np.shape[0] == latlons.shape[0], \
            f'x, y and latlon data have a different number of instances! ' \
            f'x: {x_np.shape[0]}, y: {y_np.shape[0]}, latlons: {latlons.shape[0]}'

        for idx in range(latlons.shape[0]):

            lat, lon = latlons[idx, 0], latlons[idx, 1]

            for time in range(x_np.shape[1]):
                target = x.isel(time=time).sel(lat=lat).sel(lon=lon).VHI.values

                if not normalize:
                    assert target == x_np[idx, time, 0], \
                        f'Got different x values for time idx: {time}, lat: {lat}, ' \
                        f'lon: {lon}.Expected {target}, got {x_np[idx, time, 0]}'

            target_y = y.isel(time=0).sel(lat=lat).sel(lon=lon).VHI.values
            assert target_y == y_np[idx, 0], \
                f'Got y different values for lat: {lat}, ' \
                f'lon: {lon}.Expected {target_y}, got {y_np[idx, 0]}'
