import numpy as np
import pandas as pd
import xarray as xr
from typing import List, Union, Tuple

import sys
sys.path.append("../..")

from _base_models import parsimonious, regression, linear_nn, rnn, earnn
from scripts.utils import get_data_path
from src.analysis import all_explanations_for_file
from src.analysis import read_train_data, read_test_data
from src.engineer import Engineer


def sort_by_median_target_variable(target_data: Union[xr.Dataset, xr.DataArray]) -> Tuple[np.array, pd.DataFrame]:
    target_variable = [v for v in target_data.data_vars][0]
    median_data = target_data.resample(
        time='Y').median().median(dim=['lat', 'lon'])

    median_data = median_data.to_dataframe()
    median_data["year"] = median_data.index.year

    # sorted low to high
    sorted_df = median_data.sort_values(target_variable)
    sorted_years = sorted_df.year

    return sorted_years.values, sorted_df


def get_experiment_years(sorted_years: np.array, train_length: int, hilo: str, test_length: int = 5):
    test_length = max(1, test_length)
    assert train_length >= 1
    assert hilo in ['high', 'low', 'middle']

    # split into three groups
    np.array_split(sorted_years, 3)
    return


def rename_model_experiment_file(
    data_dir: Path,
    test_years: List[int],
    train_years: List[int],
    static: bool
) -> None:
    from_path = data_dir / "models" / "one_month_forecast"
    to_path = data_dir / "models" / \
        f"one_month_forecast_{'_'.join(test_years)}_{'_'.join(train_years}"

    # with_datetime ensures that unique
    _rename_directory(from_path, to_path, with_datetime=True)


def run_all_models_as_experiments(
    test_years: List[int],
    train_years: List[int],
    static: bool,
    run_regression: bool = True,
):
    print(f"Experiment Static: {static}")

    # RUN EXPERIMENTS
    if run_regression:
        regression(ignore_vars=ignore_vars, include_static=static)

    if static:
        # 'embeddings' or 'features'
        try:
            linear_nn(ignore_vars=ignore_vars, static="embeddings")
        except RuntimeError:
            print(
                f"\n{'*'*10}\n FAILED: LinearNN \n{'*'*10}\n"
            )

        try:
            rnn(ignore_vars=ignore_vars, static="embeddings")
        except RuntimeError:
            print(
                f"\n{'*'*10}\n FAILED: RNN \n{'*'*10}\n"
            )
        try:

            earnn(pretrained=False, ignore_vars=ignore_vars, static="embeddings")
        except RuntimeError:
            print(
                f"\n{'*'*10}\n FAILED: EALSTM \n{'*'*10}\n"
            )

    else:
        try:
            linear_nn(ignore_vars=ignore_vars, static=None)
        except RuntimeError:
            print(
                f"\n{'*'*10}\n FAILED: LinearNN \n{'*'*10}\n"
            )

        try:
            rnn(ignore_vars=ignore_vars, static=None)
        except RuntimeError:
            print(
                f"\n{'*'*10}\n FAILED: RNN \n{'*'*10}\n"
            )

        try:
            earnn(pretrained=False, ignore_vars=ignore_vars, static=None)
        except RuntimeError:
            print(
                f"\n{'*'*10}\n FAILED: EALSTM \n{'*'*10}\n"
            )

    # RENAME DIRECTORY
    data_dir = get_data_path()
    rename_model_experiment_file(data_dir)
    print(f"Experiment finished")


def run_training_period_experiments():
    data_dir = get_data_path()
    target_data = xr.open_dataset(
        data_dir / 'interim' / 'VCI_preprocessed' / 'data_kenya.nc'
    )
    sorted_years, _ = sort_by_median_target_variable(
        target_data
    )

    for train_years, test_years in zip():
        engineer = Engineer(
            get_data_path(),
            experiment='one_month_forecast',
            process_static=True,
            test_year=test_years
        )

    pass


if __name__ == "__main__":
    run_training_period_experiments()
