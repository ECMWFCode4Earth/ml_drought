import sys
sys.path.append('..')

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pickle

from src.analysis import plot_shap_values
from src.models import Persistence, LinearRegression, LinearNetwork
from src.models.data import DataLoader


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
    predictor.train()
    predictor.evaluate(save_preds=True)

    # mostly to test it works
    test_arrays_loader = DataLoader(data_path=data_path, batch_file_size=1,
                                    shuffle_data=False, mode='test')
    key, val = list(next(iter(test_arrays_loader)).items())[0]

    explanations = predictor.explain(val.x)
    print(explanations.shape)
    np.save('shap_regression.npy', explanations)
    np.save('shap_x.npy', val.x)

    with open('variables.txt', 'w') as f:
        f.write(str(val.x_vars))

    # plot the variables
    with (data_path / 'features/normalizing_dict.pkl').open('rb') as f:
        normalizing_dict = pickle.load(f)

    for variable in val.x_vars:
        plt.clf()
        plot_shap_values(val.x[0], explanations[0], val.x_vars, normalizing_dict, variable,
                         normalize_shap_plots=True, show=False)
        plt.savefig(f'{variable}_linear_regression.png', dpi=300, bbox_inches='tight')


def linear_nn():
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')

    predictor = LinearNetwork(layer_sizes=[100], data_folder=data_path, experiment='one_month_forecast')
    predictor.train()
    predictor.evaluate(save_preds=True)

    # mostly to test it works
    test_arrays_loader = DataLoader(data_path=data_path, batch_file_size=1,
                                    shuffle_data=False, mode='test', to_tensor=True)
    key, val = list(next(iter(test_arrays_loader)).items())[0]

    explanations = predictor.explain(val.x[:3])
    print(explanations.shape)
    np.save('shap_linear_network.npy', explanations)
    np.save('shap_x_linear_network.npy', val.x[:3])

    with open('variables_linear_network.txt', 'w') as f:
        f.write(str(val.x_vars))

    # plot the variables
    with (data_path / 'features/normalizing_dict.pkl').open('rb') as f:
        normalizing_dict = pickle.load(f)

    for variable in val.x_vars:
        plt.clf()
        plot_shap_values(val.x[0].numpy(), explanations[0], val.x_vars, normalizing_dict, variable,
                         normalize_shap_plots=True, show=False)
        plt.savefig(f'{variable}_linear_network.png', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    parsimonious()
    regression()
    linear_nn()
