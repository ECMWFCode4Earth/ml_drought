import sys
sys.path.append('..')

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pickle

from src.analysis import plot_shap_values
from src.models import (Persistence, LinearRegression,
                        LinearNetwork, RecurrentNetwork,
                        EARecurrentNetwork)
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
    test_arrays_loader = DataLoader(data_path=data_path, batch_file_size=1,
                                    experiment=experiment,
                                    shuffle_data=False, mode='test')
    key, val = list(next(iter(test_arrays_loader)).items())[0]

    explain_hist, explain_add = predictor.explain(val.x)

    np.save('shap_regression_historical.npy', explain_hist)
    np.save('shap_regression_add.npy', explain_add)
    np.save('shap_x_hist.npy', val.x.historical)
    np.save('shap_x_add.npy', val.x.pred_months)

    with open('variables.txt', 'w') as f:
        f.write(str(val.x_vars))

    # plot the variables
    with (data_path / f'features/{experiment}/normalizing_dict.pkl').open('rb') as f:
        normalizing_dict = pickle.load(f)

    for variable in val.x_vars:
        plt.clf()
        plot_shap_values(
            val.x.historical[0], explain_hist[0], val.x_vars,
            normalizing_dict, variable, normalize_shap_plots=True,
            show=False
        )
        plt.savefig(f'{variable}_linear_regression.png', dpi=300, bbox_inches='tight')


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

    # The code below is commented out because of a bug in the shap deep Explainer which
    # prevents it from working. It has been fixed in master, but not deployed yet:
    # https://github.com/slundberg/shap/pull/684

    # test_arrays_loader = DataLoader(data_path=data_path, batch_file_size=1,
    #                                 shuffle_data=False, mode='test', to_tensor=True)
    # key, val = list(next(iter(test_arrays_loader)).items())[0]
    #
    # explain_hist, explain_add = predictor.explain([val.x.historical[:3], val.x.pred_months[:3]])
    # print(explain_hist.shape)
    # np.save('shap_linear_network_hist.npy', explain_hist)
    # np.save('shap_linear_network_add.npy', explain_add)
    #
    # np.save('shap_x_linear_network_hist.npy', val.x.historical[:3])
    # np.save('shap_x_linear_network_add.npy', val.x.pred_months[:3])
    #
    # with open('variables_linear_network.txt', 'w') as f:
    #     f.write(str(val.x_vars))
    #
    # # plot the variables
    # with (data_path / 'features/normalizing_dict.pkl').open('rb') as f:
    #     normalizing_dict = pickle.load(f)
    #
    # for variable in val.x_vars:
    #     plt.clf()
    #     plot_shap_values(val.x[0].numpy(), explain_hist[0], val.x_vars, normalizing_dict, variable,
    #                      normalize_shap_plots=True, show=False)
    #     plt.savefig(f'{variable}_linear_network.png', dpi=300, bbox_inches='tight')


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

    # See above; we need to update the shap version before this can be explained


def earnn(
    experiment='one_month_forecast',
    include_pred_month=True,
    surrounding_pixels=1
):
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')

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

    # See above; we need to update the shap version before this can be explained


if __name__ == '__main__':
    parsimonious()
    regression()
    linear_nn()
    rnn()
    earnn()
