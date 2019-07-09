import sys
sys.path.append('..')

from collections import defaultdict

from pathlib import Path
from src.models import LinearNetwork


def linear_nn_grid_search():
    def linear_nn(layer_sizes, include_pred_month=True):
        # if the working directory is alread ml_drought don't need ../data
        if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
            data_path = Path('data')
        else:
            data_path = Path('../data')

        predictor = LinearNetwork(layer_sizes=layer_sizes, data_folder=data_path,
                                  include_pred_month=include_pred_month)
        predictor.train(num_epochs=100, early_stopping=5)
        return predictor.evaluate(save_preds=True, return_total_rmse=True)

    layer_sizes_list = [[100, 100, 100], [200], [200, 200]]
    add_pred_months = [True, False]

    results_dict = defaultdict(lambda: defaultdict(float))
    for layer_sizes in layer_sizes_list:
        for pred_months in add_pred_months:
            print(f'Testing with {str(layer_sizes)} and pred months {pred_months}')
            rmse = linear_nn(layer_sizes, include_pred_month=pred_months)

            results_dict[str(layer_sizes)][str(pred_months)] = rmse

    for key, val in results_dict.items():
        for subkey, rmse in val.items():

            print(f'{key}, {subkey}: {rmse}')


if __name__ == '__main__':
    linear_nn_grid_search()
