import torch
import numpy as np
import pytest

from src.models.data import DataLoader, _BaseIter

from ..utils import _make_dataset


class TestBaseIter:

    def test_mask(self, tmp_path):

        for i in range(5):
            (tmp_path / f'features/train/{i}').mkdir(parents=True)
            (tmp_path / f'features/train/{i}/x.nc').touch()
            (tmp_path / f'features/train/{i}/y.nc').touch()

        mask_train = [True, True, False, True, False]
        mask_val = [False, False, True, False, True]

        train_paths = DataLoader._load_datasets(tmp_path, mode='train',
                                                shuffle_data=True, mask=mask_train)
        val_paths = DataLoader._load_datasets(tmp_path, mode='train',
                                              shuffle_data=True, mask=mask_val)
        assert len(set(train_paths).intersection(set(val_paths))) == 0, \
            f'Got the same file in both train and val set!'
        assert len(train_paths) + len(val_paths) == 5, f'Not all files loaded!'

    def test_pred_months(self, tmp_path):
        for i in range(1, 13):
            (tmp_path / f'features/train/2018_{i}').mkdir(parents=True)
            (tmp_path / f'features/train/2018_{i}/x.nc').touch()
            (tmp_path / f'features/train/2018_{i}/y.nc').touch()

        pred_months = [4, 5, 6]

        train_paths = DataLoader._load_datasets(tmp_path, mode='train',
                                                shuffle_data=True, pred_months=pred_months)

        assert len(train_paths) == len(pred_months), \
            f'Got {len(train_paths)} filepaths back, expected {len(pred_months)}'

        for return_file in train_paths:
            subfolder = return_file.parts[-1]
            month = int(str(subfolder)[5:])
            assert month in pred_months, f'{month} not in {pred_months}, got {return_file}'

    @pytest.mark.parametrize('normalize,to_tensor', [(True, True), (True, False),
                                                     (False, True), (False, False)])
    def test_ds_to_np(self, tmp_path, normalize, to_tensor):

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
                self.to_tensor = None
                self.surrounding_pixels = None

        base_iterator = _BaseIter(MockLoader())

        arrays = base_iterator.ds_folder_to_np(tmp_path, return_latlons=True,
                                               to_tensor=to_tensor)

        x_np, y_np, latlons = arrays.x.historical, arrays.y, arrays.latlons

        if to_tensor:
            assert (type(x_np) == torch.Tensor) and (type(y_np) == torch.Tensor)
        else:
            assert (type(x_np) == np.ndarray) and (type(y_np) == np.ndarray)

        assert x_np.shape[0] == y_np.shape[0] == latlons.shape[0], \
            f'x, y and latlon data have a different number of instances! ' \
            f'x: {x_np.shape[0]}, y: {y_np.shape[0]}, latlons: {latlons.shape[0]}'

        for idx in range(latlons.shape[0]):

            lat, lon = latlons[idx, 0], latlons[idx, 1]

            for time in range(x_np.shape[1]):
                target = x.isel(time=time).sel(lat=lat).sel(lon=lon).VHI.values

                if (not normalize) and (not to_tensor):
                    assert target == x_np[idx, time, 0], \
                        f'Got different x values for time idx: {time}, lat: {lat}, ' \
                        f'lon: {lon}.Expected {target}, got {x_np[idx, time, 0]}'

            if not to_tensor:
                target_y = y.isel(time=0).sel(lat=lat).sel(lon=lon).VHI.values
                assert target_y == y_np[idx, 0], \
                    f'Got y different values for lat: {lat}, ' \
                    f'lon: {lon}.Expected {target_y}, got {y_np[idx, 0]}'

    def test_surrounding_pixels(self):
        x, _, _ = _make_dataset(size=(10, 10))
        org_vars = list(x.data_vars)

        x_with_more = _BaseIter._add_surrounding(x, 1)
        shifted_vars = x_with_more.data_vars

        for data_var in org_vars:
            for lat in [-1, 0, 1]:
                for lon in [-1, 0, 1]:
                    if lat == lon == 0:
                        assert f'lat_{lat}_lon_{lon}_{data_var}' not in shifted_vars, \
                            f'lat_{lat}_lon_{lon}_{data_var} should not ' \
                            f'be in the shifted variables'
                    else:
                        shifted_var_name = f'lat_{lat}_lon_{lon}_{data_var}'
                        assert shifted_var_name in shifted_vars, \
                            f'{shifted_var_name} is not in the shifted variables'

                        org = x_with_more.VHI.isel(time=0, lon=5, lat=5).values
                        shifted = x_with_more[shifted_var_name].isel(time=0,
                                                                     lon=5 + lon,
                                                                     lat=5 + lat).values
                        assert org == shifted, f"Shifted values don't match!"
