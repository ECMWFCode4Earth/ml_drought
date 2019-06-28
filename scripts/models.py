import sys
sys.path.append('..')

from pathlib import Path
from src.models import Persistence, LinearRegression, LinearNetwork


def parsimonious():
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')

    predictor = Persistence(data_path, experiment='one_month_forecast')
    predictor.evaluate(save_preds=True)


def regression():
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')

    predictor = LinearRegression(data_path, experiment='one_month_forecast')
    predictor.evaluate(save_preds=True)


def linear_nn():
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')

    predictor = LinearNetwork(layer_sizes=[100], data_folder=data_path, experiment='one_month_forecast')
    predictor.evaluate(save_preds=True)


if __name__ == '__main__':
    parsimonious()
    regression()
    linear_nn()
