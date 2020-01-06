import sys

sys.path.append("../..")

from typing import List
from src.analysis import all_explanations_for_file
from scripts.utils import get_data_path
from _base_models import parsimonious, regression, linear_nn, rnn, earnn

import pandas as pd
import numpy as np

def sort_by_median_target_variable(target_variable: str = 'VCI'):

    sorted_list = df.sortby('VCI').
    return sorted_list


def get_training_years(length: int, hilo: str):
    assert hilo in ['high', 'low', 'middle']
    return


def rename_model_experiment_file(
    data_dir: Path,
    test_years: List[int],
    train_years: List[int],
    static: bool
) -> None:
    from_path = data_dir / "models" / "one_month_forecast"
    to_path = data_dir / "models" / \
        f"one_month_forecast_{}{}"

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


if __name__ == "__main__":
    pass
