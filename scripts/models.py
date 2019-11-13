import sys
sys.path.append('..')

from pathlib import Path
from src.models import (Persistence, LinearRegression,
                        LinearNetwork, RecurrentNetwork,
                        EARecurrentNetwork, load_model)
from src.analysis import all_shap_for_file

# NOTE: p84.162 == 'vertical integral of moisture flux'

def parsimonious(
    experiment='one_month_forecast',
    ignore_vars=None
):
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')

    predictor = Persistence(
        data_path, experiment=experiment, ignore_vars=ignore_vars
    )
    predictor.evaluate(save_preds=True)


def regression(
    experiment='one_month_forecast',
    include_pred_month=True,
    surrounding_pixels=None
):
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')

    predictor = LinearRegression(
        data_path, experiment=experiment,
        include_pred_month=include_pred_month,
        surrounding_pixels=surrounding_pixels,
        ignore_vars=ignore_vars,
    )
    predictor.train()
    predictor.evaluate(save_preds=True)

    # mostly to test it works
    predictor.explain(save_shap_values=True)


def linear_nn(
    experiment='one_month_forecast',
    include_pred_month=True,
    surrounding_pixels=None,
    ignore_vars=None,
    pretrained=False
):
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')
    if not pretrained:
        predictor = LinearNetwork(
            layer_sizes=[100], data_folder=data_path,
            experiment=experiment,
            include_pred_month=include_pred_month,
            surrounding_pixels=surrounding_pixels,
            ignore_vars=ignore_vars,
        )
        predictor.train(num_epochs=50, early_stopping=5)
        predictor.evaluate(save_preds=True)
        predictor.save_model()
    else:
        predictor = load_model(data_path / f'models/{experiment}/ealstm/model.pt')

    _ = predictor.explain(save_shap_values=True)


def rnn(
    experiment='one_month_forecast',
    include_pred_month=True,
    surrounding_pixels=None,
    ignore_vars=None,
    pretrained=True
):
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')
    if not pretrained:
        predictor = RecurrentNetwork(
            hidden_size=128,
            data_folder=data_path,
            experiment=experiment,
            include_pred_month=include_pred_month,
            surrounding_pixels=surrounding_pixels,
            ignore_vars=ignore_vars,
        )
        predictor.train(num_epochs=50, early_stopping=5)
        predictor.evaluate(save_preds=True)
        predictor.save_model()
    else:
        predictor = load_model(data_path / f'models/{experiment}/rnn/model.pt')

    _ = predictor.explain(save_shap_values=True)


def earnn(
    experiment='one_month_forecast',
    include_pred_month=True,
    surrounding_pixels=None,
    pretrained=True,
    ignore_vars=None
):
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')

    if not pretrained:
        predictor = EARecurrentNetwork(
            hidden_size=128,
            data_folder=data_path,
            experiment=experiment,
            include_pred_month=include_pred_month,
            surrounding_pixels=surrounding_pixels,
            ignore_vars=ignore_vars,
        )
        predictor.train(num_epochs=50, early_stopping=5)
        predictor.evaluate(save_preds=True)
        predictor.save_model()
    else:
        predictor = load_model(data_path / f'models/{experiment}/ealstm/model.pt')

    test_file = data_path / f'features/{experiment}/test/2018_3'
    assert test_file.exists()
    all_shap_for_file(test_file, predictor, batch_size=100)


if __name__ == '__main__':
    ignore_vars = None
    ignore_vars = ['VCI', 'p84.162', 'sp', 'tp']

    # parsimonious(ignore_vars=ignore_vars)
    # regression(ignore_vars=ignore_vars)
    # linear_nn(ignore_vars=ignore_vars)
    # rnn(ignore_vars=ignore_vars)
    earnn(pretrained=False, ignore_vars=ignore_vars)
