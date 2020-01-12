import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import itertools
from typing import List, Union, Tuple, Dict, Optional

import sys

sys.path.append("../..")

from scripts.utils import get_data_path, _rename_directory
from src.engineer import Engineer
from src.analysis import read_train_data, read_test_data
from src.analysis import all_explanations_for_file
from src.engineer import Engineer
from _base_models import parsimonious, regression, linear_nn, rnn, earnn


def sort_by_median_target_variable(
    target_data: Union[xr.Dataset, xr.DataArray]
) -> Tuple[np.array, pd.DataFrame]:
    target_variable = [v for v in target_data.data_vars][0]
    median_data = target_data.resample(time="Y").median().median(dim=["lat", "lon"])

    median_data = median_data.to_dataframe()
    median_data["year"] = median_data.index.year

    # sorted low to high
    sorted_df = median_data.sort_values(target_variable)
    sorted_years = sorted_df.year

    return sorted_years.values, sorted_df


def _calculate_hilo_dict(sorted_years: np.array) -> Dict[str, np.array]:
    # split into three groups
    low, med, high = tuple(np.array_split(sorted_years, 3))
    hilo_dict = {"low": low, "med": med, "high": high}

    return hilo_dict


def get_experiment_years(
    sorted_years: np.array,
    train_length: int,
    test_hilo: str,
    train_hilo: str,
    test_length: int = 3,
) -> Tuple[np.array, np.array]:
    test_length = max(1, test_length)
    assert train_length >= 1, "Must have at least one year for training"
    assert train_length <= len(sorted_years) - test_length, (
        "Cannot have more test_years than total - test_length"
        f"{train_length} > {len(sorted_years) - test_length}"
    )
    assert train_hilo in [
        "high",
        "low",
        "med",
    ], "hilo must be one of ['high', 'low', 'med']"
    assert test_hilo in [
        "high",
        "low",
        "med",
    ], "hilo must be one of ['high', 'low', 'med']"

    # 1. split into low, med, high groups
    test_dict = _calculate_hilo_dict(sorted_years)

    # 2. select randomly the TEST years from these groups
    test_years = np.random.choice(test_dict[test_hilo], test_length, replace=False)
    # test_indexes = np.array(
    #     [np.where(test_dict[test_hilo] == i) for i in test_years]
    # ).flatten()

    # 3. remove test_years from array to choose train_years
    # NOTE: do we reshuffle or take from the already prescribed groups?
    reshuffle = True
    if reshuffle:
        sorted_years_for_training = np.array(
            [yr for yr in sorted_years if yr not in test_years]
        )
        train_dict = _calculate_hilo_dict(sorted_years_for_training)

    else:
        train_dict = test_dict.copy()
        train_dict[train_hilo] = np.array(
            [yr for yr in train_dict[train_hilo] if yr not in test_years]
        )

    # 4. Choose the train_years
    # NOTE: do we sample with replacement?
    replacement = False
    # get the n training years from the group
    if train_length > len(train_dict[train_hilo]):
        # if the number of training years is more than the group
        # we need to 'steal' some years from other groups
        train_years = train_dict[train_hilo]

        leftover_years = [
            i for i in sorted_years if (i not in train_years) & (i not in test_years)
        ]

        # iteratively split the array into low, med, high and keep selecting
        # test_years from the correct group until they're all used up
        while len(train_years) < train_length:
            # a. no more years left to select
            if len(leftover_years) == 0:
                print("No more years left for training")
                break

            # calculate a new lo,med,hi dictionary
            train_dict = _calculate_hilo_dict(leftover_years)

            # if we run out of years in our chosen group
            # e.g. {low=[1991], med=[1992], high=[]}
            # and we want high training years, select from other groups
            if len(train_dict[train_hilo]) == 0:
                print("not enough {train_hilo} years left! Selecting from other groups")
                train_years = np.append(
                    train_years, np.random.choice(leftover_years, 1)
                )
                leftover_years = [
                    i
                    for i in sorted_years
                    if (i not in train_years) & (i not in test_years)
                ]
                assert False, "need to select the leftovers in a neater way?"
                continue
            train_years = np.append(train_years, train_dict[train_hilo])
            leftover_years = [
                i
                for i in sorted_years
                if (i not in train_years) & (i not in test_years)
            ]

    else:
        train_years = np.random.choice(
            train_dict[train_hilo], train_length, replace=replacement
        )

    # train_indexes = np.array(
    #     [np.where(sorted_years == i) for i in train_years]
    # ).flatten()

    return test_years, train_years


def rename_experiment_dir(
    data_dir: Path,
    train_hilo: str,
    test_hilo: str,
    train_length: int,
    dir_: str = "models",
) -> None:
    from_path = data_dir / dir_ / "one_month_forecast"

    to_path = (
        data_dir
        / dir_
        / f"one_month_forecast_TR{train_hilo}_TE{test_hilo}_LEN{train_length}"
    )

    # with_datetime ensures that unique
    _rename_directory(from_path, to_path, with_datetime=True)


def run_experiments(
    train_hilo: str,
    test_hilo: str,
    train_length: int,
    static: bool,
    ignore_vars: Optional[List[str]] = None,
    run_regression: bool = True,
    all_models: bool = False,
):
    # run baseline model
    print("\n\nBASELINE MODEL:")
    parsimonious()
    print("\n\n")

    # RUN EXPERIMENTS
    if run_regression:
        regression(ignore_vars=ignore_vars, include_static=static)

    if static:
        # 'embeddings' or 'features'
        try:
            earnn(pretrained=False, ignore_vars=ignore_vars, static="embeddings")
        except RuntimeError:
            print(f"\n{'*'*10}\n FAILED: EALSTM \n{'*'*10}\n")

        if all_models:  # run all other models ?
            try:
                linear_nn(ignore_vars=ignore_vars, static="embeddings")
            except RuntimeError:
                print(f"\n{'*'*10}\n FAILED: LinearNN \n{'*'*10}\n")

            try:
                rnn(ignore_vars=ignore_vars, static="embeddings")
            except RuntimeError:
                print(f"\n{'*'*10}\n FAILED: RNN \n{'*'*10}\n")

    else:  # NO STATIC data
        try:
            rnn(ignore_vars=ignore_vars, static=None)
        except RuntimeError:
            print(f"\n{'*'*10}\n FAILED: RNN \n{'*'*10}\n")

        if all_models:  # run all other models ?
            try:
                linear_nn(ignore_vars=ignore_vars, static=None)
            except RuntimeError:
                print(f"\n{'*'*10}\n FAILED: LinearNN \n{'*'*10}\n")

    # RENAME DIRECTORY
    data_dir = get_data_path()
    rename_experiment_dir(
        data_dir, train_hilo=train_hilo, test_hilo=test_hilo, train_length=train_length
    )
    print(f"Experiment finished")


class Experiment:
    """
    train_length: int
        the length of the training period (# years)
    test_length: int = 3
        the length of the testing period (# years)
    train_hilo: str
        selecting the training years from which tercile?
        one of ['high', 'med', 'low']
    test_hilo: str
        selecting the training years from which tercile?
        one of ['high', 'med', 'low']

    @dataclass
    train_length: int
    test_length: int = 3
    train_hilo: str
    test_hilo: str
    """

    def __init__(
        self, train_length: int, train_hilo: str, test_hilo: str, test_length: int = 3
    ):
        self.train_length = train_length
        self.train_hilo = train_hilo
        self.test_hilo = test_hilo
        self.test_length = test_length

        assert train_hilo in ["high", "med", "low"]
        assert test_hilo in ["high", "med", "low"]


def run_training_period_experiments(pred_months: int = 3):
    expected_length = pred_months

    # Read the target data
    print("** Reading the target data **")
    data_dir = get_data_path()
    target_data = xr.open_dataset(
        data_dir / "interim" / "VCI_preprocessed" / "data_kenya.nc"
    )
    # sort by the annual median (across pixels/time)
    print("** Sorting the target data **")
    sorted_years, _ = sort_by_median_target_variable(target_data)
    print(f"** sorted_years: {sorted_years} **")
    print(f"** min_year: {min(sorted_years)} max_year: {max(sorted_years)} **")

    # create all experiments
    # train_hilo(9), test_hilo(3), train_length(1)
    print("** Creating all experiments **")
    hilos = ["high", "med", "low"]
    train_lengths = [5, 10, 20]
    experiments = [
        Experiment(
            train_length=train_length, train_hilo=train_hilo, test_hilo=test_hilo
        )
        for train_hilo, test_hilo, train_length in itertools.product(
            hilos, hilos, train_lengths
        )
    ]

    print("** Running all experiments **")
    for experiment in experiments[4:]:
        test_years, train_years = get_experiment_years(
            sorted_years,
            experiment.train_length,
            experiment.test_hilo,
            experiment.train_hilo,
            test_length=3,
        )

        debug = True
        if debug:
            print(
                "\n" + "-" * 10 + "\n",
                "train_length: " + str(experiment.train_length),
                "test_hilo: " + experiment.test_hilo,
                "train_hilo: " + experiment.train_hilo,
                "\ntrain_years:\n",
                train_years,
                "\n",
                "test_years:\n",
                test_years,
                "\n" + "-" * 10 + "\n",
            )

        # have to recreate each engineer for the experiment
        # TODO: definite inefficiency should this be in DataLoader?
        engineer = Engineer(
            get_data_path(),
            experiment="one_month_forecast",
            process_static=True,
            different_training_periods=True,
        )
        engineer.engineer_class.engineer(
            test_year=test_years,  # defined by experiment
            train_years=train_years,  # defined by experiment
            pred_months=pred_months,  # 3 by default
            expected_length=expected_length,  # == pred_month by default
            target_variable="VCI",
        )

        # TODO:
        # add extra years if selected the first year in timeseries (often not 12months)
        # e.g. 1981_11 is the first valid month in our dataset

        # Run the models
        always_ignore_vars = ["ndvi", "p84.162", "sp", "tp", "Eb"]
        ignore_vars = always_ignore_vars
        run_experiments(
            train_hilo=experiment.train_hilo,
            test_hilo=experiment.test_hilo,
            train_length=len(train_years),
            ignore_vars=ignore_vars,
            run_regression=False,
            all_models=False,
            static=True,
        )

        # rename the features/one_month_forecast directory
        rename_experiment_dir(
            data_dir,
            train_hilo=experiment.train_hilo,
            test_hilo=experiment.test_hilo,
            train_length=len(train_years),
            dir_="features",
        )


if __name__ == "__main__":
    pred_months = 3

    run_training_period_experiments(pred_months=pred_months)
