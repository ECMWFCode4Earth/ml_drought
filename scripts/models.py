import sys
sys.path.append('..')

from pathlib import Path

from src.models import Persistence, LinearRegression, LinearNetwork
from src.models.data import DataLoader


def parsimonious():
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')

    predictor = Persistence(data_path)
    predictor.evaluate(save_preds=True)


def regression():
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')

    predictor = LinearRegression(data_path)
    predictor.evaluate(save_preds=True)

    # mostly to test it works
    test_arrays_loader = DataLoader(data_path=data_path, batch_file_size=1,
                                    shuffle_data=False, mode='test')
    key, val = list(next(iter(test_arrays_loader)).items())[0]
    explanations = predictor.explain(val.x)
    print(explanations.shape)


def linear_nn():
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')

    predictor = LinearNetwork(layer_sizes=[100], data_folder=data_path)
    predictor.evaluate(save_preds=True)


if __name__ == '__main__':
    # parsimonious()
    regression()
    linear_nn()
