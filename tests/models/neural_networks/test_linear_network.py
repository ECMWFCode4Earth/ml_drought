import pickle
from copy import copy
import pytest
import xarray as xr

from src.models import LinearNetwork
from src.models.neural_networks.linear_network import LinearModel

from tests.utils import _make_dataset


class TestLinearNetwork:

    def test_save(self, tmp_path, monkeypatch):

        layer_sizes = [10]
        input_layer_sizes = copy(layer_sizes)
        input_size = 10
        dropout = 0.25
        include_pred_month = True
        include_latlons = True
        include_monthly_aggs = True
        surrounding_pixels = 1
        ignore_vars = ['precip']
        include_yearly_aggs = True

        def mocktrain(self):
            self.model = LinearModel(
                input_size, layer_sizes, dropout, include_pred_month,
                include_latlons, include_yearly_aggs, include_static=True
            )
            self.input_size = input_size

        monkeypatch.setattr(LinearNetwork, 'train', mocktrain)

        model = LinearNetwork(data_folder=tmp_path, layer_sizes=layer_sizes,
                              dropout=dropout, experiment='one_month_forecast',
                              include_pred_month=include_pred_month,
                              include_latlons=include_latlons,
                              include_monthly_aggs=include_monthly_aggs,
                              include_yearly_aggs=include_yearly_aggs,
                              surrounding_pixels=surrounding_pixels,
                              ignore_vars=ignore_vars)
        model.train()
        model.save_model()

        assert (
            tmp_path / 'models/one_month_forecast/linear_network/model.pkl'
        ).exists(), f'Model not saved!'

        with (model.model_dir / 'model.pkl').open('rb') as f:
            model_dict = pickle.load(f)

        for key, val in model_dict['model']['state_dict'].items():
            assert (model.model.state_dict()[key] == val).all()

        assert model_dict['dropout'] == dropout
        assert model_dict['layer_sizes'] == input_layer_sizes
        assert model_dict['model']['input_size'] == input_size
        assert model_dict['include_pred_month'] == include_pred_month
        assert model_dict['include_latlons'] == include_latlons
        assert model_dict['include_monthly_aggs'] == include_monthly_aggs
        assert model_dict['include_yearly_aggs'] == include_yearly_aggs
        assert model_dict['surrounding_pixels'] == surrounding_pixels
        assert model_dict['ignore_vars'] == ignore_vars

    @pytest.mark.parametrize(
        'use_pred_months,use_latlons,experiment,monthly_agg,static',
        [(True, False, 'one_month_forecast', True, False),
         (False, True, 'one_month_forecast', False, True),
         (False, True, 'nowcast', True, False),
         (True, False, 'nowcast', False, True)]
    )
    def test_train(self, tmp_path, capsys, use_pred_months, use_latlons, experiment,
                   monthly_agg, static):
        # make the x, y data (5*5 latlons, 36 timesteps, 3 features)
        x, _, _ = _make_dataset(size=(5, 5), const=True)
        y = x.isel(time=[-1])

        x_add1, _, _ = _make_dataset(size=(5, 5), const=True, variable_name='precip')
        x_add2, _, _ = _make_dataset(size=(5, 5), const=True, variable_name='temp')
        x = xr.merge([x, x_add1, x_add2])

        norm_dict = {'VHI': {'mean': 0, 'std': 1},
                     'precip': {'mean': 0, 'std': 1},
                     'temp': {'mean': 0, 'std': 1}}

        test_features = tmp_path / f'features/{experiment}/train/hello'
        test_features.mkdir(parents=True, exist_ok=True)

        # make the normalising dictionary
        with (
            tmp_path / f'features/{experiment}/normalizing_dict.pkl'
        ).open('wb') as f:
            pickle.dump(norm_dict, f)

        x.to_netcdf(test_features / 'x.nc')
        y.to_netcdf(test_features / 'y.nc')

        if static:
            x_static, _, _ = _make_dataset(size=(5, 5), add_times=False)
            static_features = tmp_path / f'features/static'
            static_features.mkdir(parents=True)
            x_static.to_netcdf(static_features / 'data.nc')

            static_norm_dict = {'VHI': {'mean': 0.0, 'std': 1.0}}
            with (
                tmp_path / f'features/static/normalizing_dict.pkl'
            ).open('wb') as f:
                pickle.dump(static_norm_dict, f)

        layer_sizes = [10]
        dropout = 0.25

        model = LinearNetwork(data_folder=tmp_path, layer_sizes=layer_sizes,
                              dropout=dropout, experiment=experiment,
                              include_pred_month=use_pred_months,
                              include_latlons=use_latlons,
                              include_monthly_aggs=monthly_agg,
                              include_static=static)

        model.train()

        # check the number of input features is properly initialised
        n_input_features = [
            p for p in model.model.dense_layers.parameters()
        ][0].shape[-1]

        # Expect to have 12 more features if use_pred_months
        if experiment == 'nowcast':
            n_expected = 107
        else:
            # NOTE: data hasn't been through `src.Engineer` therefore including
            #  current data (hence why more features than `nowcast`)
            n_expected = 108

        if monthly_agg:
            n_expected *= 2
        if use_pred_months:
            n_expected += 12
        if use_latlons:
            n_expected += 2

        n_expected += 3  # +3 for the yearly means

        if static:
            n_expected += 1  # for the static variable

        assert n_input_features == n_expected, "Expected the number" \
            f"of input features to be: {n_expected}" \
            f"Got: {n_input_features}"

        captured = capsys.readouterr()
        expected_stdout = 'Epoch 1, train smooth L1: '
        assert expected_stdout in captured.out

        assert type(model.model) == LinearModel, \
            f'Model attribute not a linear regression!'

    @pytest.mark.parametrize(
        'use_pred_months,use_latlons,experiment',
        [(True, True, 'one_month_forecast'),
         (True, False, 'one_month_forecast'),
         (False, True, 'nowcast'),
         (False, False, 'nowcast')]
    )
    def test_predict(self, tmp_path, use_pred_months, use_latlons, experiment):
        x, _, _ = _make_dataset(size=(5, 5), const=True)
        y = x.isel(time=[-1])

        train_features = tmp_path / f'features/{experiment}/train/hello'
        train_features.mkdir(parents=True)

        test_features = tmp_path / f'features/{experiment}/test/hello'
        test_features.mkdir(parents=True)

        # static
        x_static, _, _ = _make_dataset(size=(5, 5), add_times=False)
        static_features = tmp_path / f'features/static'
        static_features.mkdir(parents=True)
        x_static.to_netcdf(static_features / 'data.nc')

        static_norm_dict = {'VHI': {'mean': 0.0, 'std': 1.0}}
        with (
            tmp_path / f'features/static/normalizing_dict.pkl'
        ).open('wb') as f:
            pickle.dump(static_norm_dict, f)

        # if nowcast we need another x feature
        if experiment == 'nowcast':
            x_add1, _, _ = _make_dataset(size=(5, 5), const=True, variable_name='precip')
            x_add2, _, _ = _make_dataset(size=(5, 5), const=True, variable_name='temp')
            x = xr.merge([x, x_add1, x_add2])

            norm_dict = {'VHI': {'mean': 0, 'std': 1},
                         'precip': {'mean': 0, 'std': 1},
                         'temp': {'mean': 0, 'std': 1}}
        else:
            norm_dict = {'VHI': {'mean': 0, 'std': 1}}

        with (
            tmp_path / f'features/{experiment}/normalizing_dict.pkl'
        ).open('wb') as f:
            pickle.dump(norm_dict, f)

        x.to_netcdf(test_features / 'x.nc')
        y.to_netcdf(test_features / 'y.nc')

        x.to_netcdf(train_features / 'x.nc')
        y.to_netcdf(train_features / 'y.nc')

        layer_sizes = [10]
        dropout = 0.25

        model = LinearNetwork(data_folder=tmp_path,
                              layer_sizes=layer_sizes,
                              dropout=dropout,
                              experiment=experiment,
                              include_pred_month=use_pred_months,
                              include_latlons=use_latlons)
        model.train()
        test_arrays_dict, pred_dict = model.predict()

        # the foldername "hello" is the only one which should be in the dictionaries
        assert ('hello' in test_arrays_dict.keys()) and (
            len(test_arrays_dict) == 1
        )
        assert ('hello' in pred_dict.keys()) and (len(pred_dict) == 1)

        # _make_dataset with const=True returns all ones
        assert (test_arrays_dict['hello']['y'] == 1).all()

    def test_get_background(self, tmp_path):
        x, _, _ = _make_dataset(size=(5, 5), const=True)
        y = x.isel(time=[-1])

        train_features = tmp_path / 'features/one_month_forecast/train/hello'
        train_features.mkdir(parents=True)

        x.to_netcdf(train_features / 'x.nc')
        y.to_netcdf(train_features / 'y.nc')

        norm_dict = {'VHI': {'mean': 0, 'std': 1}}
        with (
            tmp_path / 'features/one_month_forecast/normalizing_dict.pkl'
        ).open('wb') as f:
            pickle.dump(norm_dict, f)

        # static
        x_static, _, _ = _make_dataset(size=(5, 5), add_times=False)
        static_features = tmp_path / f'features/static'
        static_features.mkdir(parents=True)
        x_static.to_netcdf(static_features / 'data.nc')

        static_norm_dict = {'VHI': {'mean': 0.0, 'std': 1.0}}
        with (
            tmp_path / f'features/static/normalizing_dict.pkl'
        ).open('wb') as f:
            pickle.dump(static_norm_dict, f)

        model = LinearNetwork(data_folder=tmp_path, layer_sizes=[100],
                              dropout=0.25, include_pred_month=True)
        background = model._get_background(sample_size=3)
        assert background[0].shape[0] == 3, \
            f'Got {background[0].shape[0]} samples back, expected 3'
        assert background[1].shape[0] == 3, \
            f'Got {background[1].shape[0]} samples back, expected 3'
        assert len(background[1].shape) == 2, \
            f'Expected 2 dimensions, got {len(background[1].shape)}'
