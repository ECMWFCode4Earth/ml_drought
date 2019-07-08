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

    predictor = Persistence(data_path)
    predictor.evaluate(save_preds=True)


def regression():
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')

    predictor = LinearRegression(data_path)
    predictor.train()
    predictor.evaluate(save_preds=True)

    # mostly to test it works
    test_arrays_loader = DataLoader(data_path=data_path, batch_file_size=1,
                                    shuffle_data=False, mode='test')
    key, val = list(next(iter(test_arrays_loader)).items())[0]

    explain_hist, explain_add = predictor.explain([val.x.historical, val.x.additional])
    print(explain_hist.shape)
    np.save('shap_regression_historical.npy', explain_hist)
    np.save('shap_regression_add.npy', explain_add)
    np.save('shap_x_hist.npy', val.x.historical)
    np.save('shap_x_add.npy', val.x.additional)

    with open('variables.txt', 'w') as f:
        f.write(str(val.x_vars))

    # plot the variables
    with (data_path / 'features/normalizing_dict.pkl').open('rb') as f:
        normalizing_dict = pickle.load(f)

    for variable in val.x_vars:
        plt.clf()
        plot_shap_values(val.x.historical[0], explain_hist[0], val.x_vars, normalizing_dict,
                         variable, normalize_shap_plots=True, show=False)
        plt.savefig(f'{variable}_linear_regression.png', dpi=300, bbox_inches='tight')


def linear_nn():
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')

    predictor = LinearNetwork(layer_sizes=[100], data_folder=data_path)
    predictor.train()
    predictor.evaluate(save_preds=True)

    # mostly to test it works
    test_arrays_loader = DataLoader(data_path=data_path, batch_file_size=1,
                                    shuffle_data=False, mode='test', to_tensor=True)
    key, val = list(next(iter(test_arrays_loader)).items())[0]

    explain_hist, explain_add = predictor.explain([val.x.historical[:3], val.x.additional[:3]])
    print(explain_hist.shape)
    np.save('shap_linear_network_hist.npy', explain_hist)
    np.save('shap_linear_network_add.npy', explain_add)

    np.save('shap_x_linear_network_hist.npy', val.x.historical[:3])
    np.save('shap_x_linear_network_add.npy', val.x.additional[:3])

    with open('variables_linear_network.txt', 'w') as f:
        f.write(str(val.x_vars))

    # plot the variables
    with (data_path / 'features/normalizing_dict.pkl').open('rb') as f:
        normalizing_dict = pickle.load(f)

    for variable in val.x_vars:
        plt.clf()
        plot_shap_values(val.x[0].numpy(), explain_hist[0], val.x_vars, normalizing_dict, variable,
                         normalize_shap_plots=True, show=False)
        plt.savefig(f'{variable}_linear_network.png', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    parsimonious()
    regression()
    linear_nn()
