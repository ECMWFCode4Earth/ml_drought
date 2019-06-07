import numpy as np

from src.models.base import ModelBase, ModelArrays

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

        arrays = ModelBase.ds_folder_to_np(tmp_path, return_latlons=True)

        x_np, y_np, latlons = arrays.x, arrays.y, arrays.latlons

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

    def test_evaluate(self, tmp_path, monkeypatch, capsys):

        def mockreturn(self):

            x = np.array([1, 1, 1, 1, 1])
            y = np.array([1, 1, 1, 1, 1])

            test_arrays = {'hello': ModelArrays(x=x, y=y, x_vars=['foo'],
                                                y_var='bar')}
            preds_arrays = {'hello': y}

            return test_arrays, preds_arrays

        monkeypatch.setattr(ModelBase, 'predict', mockreturn)

        base = ModelBase(tmp_path)
        base.evaluate(save_results=False, save_preds=False)

        captured = capsys.readouterr()
        expected_stdout = 'RMSE: 0.0'
        assert expected_stdout in captured.out, \
            f'Expected stdout to be {expected_stdout}, got {captured.out}'

    def test_normalizing(self, tmp_path):

        base = ModelBase(tmp_path)
        base.model_dir = base.models_dir / 'base'
        base.model_dir.mkdir(parents=True)

        # batch_size = 100, timesteps = 11, n_features = 2
        array_shape = (100, 11, 2)
        test_array = np.ones(array_shape)

        normalized = base.normalize(test_array, mode='train')

        assert normalized.shape == array_shape, f'Normalizing changed the shape of the array!'
        assert base.normalizing_values is not None, \
            f'Normalizing values not assigned in the model!'
        assert (base.normalizing_values['mean'] == np.mean(test_array, axis=0)).all()
        assert (base.normalizing_values['std'] == np.std(test_array, axis=0)).all()

        # finally, we check its been saved
        assert (base.model_dir / 'normalizing_values.pkl').exists(), \
            f'Normalizing dict not saved!'

        # then, we check it loads properly
        expected_vals = base.normalizing_values
        base.normalizing_values = None
        assert base.normalizing_values is None  # This is just to make sure the next test works

        base.load_normalizing_values()
        for key, val in expected_vals.items():
            assert (base.normalizing_values[key] == val).all(), \
                f'Normalizing dict not properly loaded!'
