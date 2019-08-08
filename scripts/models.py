import sys
sys.path.append('..')

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pickle

from src.analysis import plot_shap_values
from src.models import (Persistence, LinearRegression,
                        LinearNetwork, RecurrentNetwork,
                        EARecurrentNetwork, load_model)
from src.models.data import DataLoader


def parsimonious(
    experiment='one_month_forecast',
):
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')

    predictor = Persistence(data_path, experiment=experiment)
    predictor.evaluate(save_preds=True)


def regression(
    experiment='one_month_forecast',
    include_pred_month=True,
    surrounding_pixels=1
):
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')

    predictor = LinearRegression(
        data_path, experiment=experiment,
        include_pred_month=include_pred_month,
        surrounding_pixels=surrounding_pixels
    )
    predictor.train()
    predictor.evaluate(save_preds=True)

    # mostly to test it works
    predictor.explain(save_shap_values=True)


def linear_nn(
    experiment='one_month_forecast',
    include_pred_month=True,
    surrounding_pixels=1
):
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')

    predictor = LinearNetwork(
        layer_sizes=[100], data_folder=data_path,
        experiment=experiment,
        include_pred_month=include_pred_month,
        surrounding_pixels=surrounding_pixels
    )
    predictor.train(num_epochs=50, early_stopping=5)
    predictor.evaluate(save_preds=True)
    predictor.save_model()

    _ = predictor.explain(save_shap_values=True)


def rnn(
    experiment='one_month_forecast',
    include_pred_month=True,
    surrounding_pixels=1
):
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')

    predictor = RecurrentNetwork(
        hidden_size=128,
        data_folder=data_path,
        experiment=experiment,
        include_pred_month=include_pred_month,
        surrounding_pixels=surrounding_pixels
    )
    predictor.train(num_epochs=50, early_stopping=5)
    predictor.evaluate(save_preds=True)
    predictor.save_model()

    _ = predictor.explain(save_shap_values=True)


def earnn(
    experiment='one_month_forecast',
    include_pred_month=True,
    surrounding_pixels=1,
    pretrained=True
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
            surrounding_pixels=surrounding_pixels
        )
        predictor.train(num_epochs=50, early_stopping=5)
        predictor.evaluate(save_preds=True)
        predictor.save_model()
    else:
        predictor = load_model(data_path / f'models/{experiment}/ealstm/model.pkl')

    _ = predictor.explain(save_shap_values=True)


if __name__ == '__main__':
    # parsimonious()
    # regression()
    # linear_nn()
    # rnn()
    earnn(pretrained=True)
