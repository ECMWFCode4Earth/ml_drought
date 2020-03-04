from pathlib import Path
from typing import List, Tuple, Dict
import sys
import itertools
import pandas as pd
import numpy as np
import xarray as xr
import json
import matplotlib.pyplot as plt

sys.path.append("../..")

from _base_models import parsimonious, regression, linear_nn, rnn, earnn
from experiment import (
    make_monthly_calendar_plot,
    calculate_length_of_hi_med_lo_experiment_train_years,
    get_valid_test_timesteps,
    sort_by_median_target_var,
    Experiment,
)

from scripts.utils import _rename_directory, get_data_path
from src.engineer import Engineer
import calendar


def lstm(ignore_vars, static):
    rnn(
        experiment="one_month_forecast",
        surrounding_pixels=None,
        explain=False,
        ignore_vars=ignore_vars,
        num_epochs=50,
        early_stopping=10,
        hidden_size=256,
        # static data
        static=static,
        include_pred_month=True if static is not None else False,
        include_latlons=True if static is not None else False,
        include_prev_y=True if static is not None else False,
    )


def ealstm(ignore_vars, static="features"):
    # -------------
    # EALSTM
    # -------------
    earnn(
        experiment="one_month_forecast",
        surrounding_pixels=None,
        pretrained=False,
        explain=False,
        ignore_vars=ignore_vars,
        num_epochs=50,
        early_stopping=10,
        hidden_size=256,
        static_embedding_size=64,
        # static data
        static=static,
        include_pred_month=True,
        include_latlons=True,
        include_prev_y=True,
    )


def rename_experiment_dir(
    data_dir: Path,
    train_hilo: str,
    test_hilo: str,
    train_length: int,
    dir_: str = "models",
    with_datetime: bool = False,
) -> Path:
    from_path = data_dir / dir_ / "one_month_forecast"

    to_path = (
        data_dir
        / dir_
        / f"one_month_forecast_TR{train_hilo}_TE{test_hilo}_LEN{train_length}"
    )

    # with_datetime ensures that unique
    _rename_directory(from_path, to_path, with_datetime=with_datetime)

    return to_path


def run_experiments(
    vars_to_exclude: List,
    pred_timesteps: int = 3,
    test_length: int = 12,
    target_var="boku_VCI",
    data_dir: Path = Path("data"),
):
    expected_length = pred_timesteps

    # 1. Read the target data
    print(f"** Reading the target data for {target_var}**")

    if target_var == "VCI":
        target_data = xr.open_dataset(
            data_dir / "interim" / "VCI_preprocessed" / "data_kenya.nc"
        )

    if target_var == "boku_VCI":
        target_data = xr.open_dataset(
            data_dir / "interim" / "boku_ndvi_1000_preprocessed" / "data_kenya.nc"
        )

    target_data = target_data[[target_var]]
    hilos = ["high", "med", "low"]

    #  2. Sort the target data by MEDIAN MONTH values
    sorted_df, sorted_timesteps = sort_by_median_target_var(
        pred_timesteps=pred_timesteps, data_dir=data_dir
    )

    # 3. Calculate the train lengths ([1/3, 2/3, 3/3] of leftover data after test)
    train_lengths = calculate_length_of_hi_med_lo_experiment_train_years(
        total_months=len(sorted_timesteps),
        test_length=test_length,
        pred_timesteps=pred_timesteps,
    )

    # 4. create all experiments
    # train_hilo(9), test_hilo(3), train_length(1)
    print("** Creating all experiments **")
    experiments = [
        Experiment(
            train_length=train_length,
            train_hilo=train_hilo,
            test_hilo=test_hilo,
            test_length=test_length,
            sorted_timesteps=sorted_timesteps,
        )
        for train_hilo, test_hilo, train_length in itertools.product(
            hilos, hilos, train_lengths
        )
    ]

    # 5. Run the experiments
    print("** Running all experiments **")
    for experiment in experiments:
        test_timesteps, train_timesteps = (
            experiment.test_timesteps,
            experiment.train_timesteps,
        )
        assert len(test_timesteps) == int(experiment.test_length), (
            f"Expect the number of test timesteps to be: {experiment.test_length}"
            f"Got: {len(test_timesteps)}"
        )

        if DEBUG:
            experiment.print_experiment_summary()

        # a. Run the Engineer for these train/test periods
        engineer = Engineer(
            get_data_path(),
            experiment="one_month_forecast",
            process_static=True,
            different_training_periods=True,
        )
        engineer.engineer_class.engineer(
            train_timesteps=[
                pd.to_datetime(t) for t in train_timesteps
            ],  # defined by experiment
            test_timesteps=[
                pd.to_datetime(t) for t in test_timesteps
            ],  # defined by experiment
            pred_months=pred_timesteps,  # 3 by default
            expected_length=expected_length,  # == pred_month by default
            target_variable=target_var,
            train_years=None,
            test_year=None,
        )

        test_nc_files = [
            d.name for d in (data_dir / "features/one_month_forecast/test").iterdir()
        ]
        test_nc_files.sort()
        assert len(test_nc_files) == int(experiment.test_length), (
            f"Expect the Engineer to have created {experiment.test_length} files "
            f"Got: {len(test_nc_files)} \n\n {test_nc_files}"
        )

        #  b. run the models
        parsimonious()
        lstm(vars_to_exclude, static="features")
        ealstm(vars_to_exclude, static="features")

        # c. rename the directories (TRAIN/TEST)
        data_dir = get_data_path()
        features_path = rename_experiment_dir(
            data_dir,
            train_hilo=experiment.train_hilo,
            test_hilo=experiment.test_hilo,
            train_length=experiment.train_length,
            dir_="features",
            with_datetime=True,
        )
        models_path = rename_experiment_dir(
            data_dir,
            train_hilo=experiment.train_hilo,
            test_hilo=experiment.test_hilo,
            train_length=experiment.train_length,
            dir_="models",
            with_datetime=True,
        )

        # d. save the experiment metadata
        save_object = dict(
            train_hilo=experiment.train_hilo,
            test_hilo=experiment.test_hilo,
            train_length=len(experiment.train_timesteps),
            ignore_vars=vars_to_exclude,
            static=static,
            train_timesteps=experiment.train_timesteps,
            test_timesteps=experiment.test_timesteps,
            features_path=features_path,
            models_path=models_path,
            sorted_timesteps=experiment.sorted_timesteps,
        )

        # with open(models_path / "experiment.json", "wb") as fp:
        #     json.dump(save_object, fp, sort_keys=True, indent=4)


def move_all_experiment_files(data_dir: Path, folder_name: str, dir_: str = "models"):
    assert dir_ in ["models", "features"]

    all_experiment_dirs = [d for d in (data_dir / dir_).glob("*TR*TE*LEN*")]

    if not (data_dir / dir_ / folder_name).exists():
        (data_dir / dir_ / folder_name).mkdir(exist_ok=True, parents=True)

    for from_path in all_experiment_dirs:
        to_path = data_dir / dir_ / folder_name / from_path.name
        _rename_directory(from_path=from_path, to_path=to_path)


if __name__ == "__main__":
    global DEBUG
    DEBUG = True

    all_vars = [
        "VCI",
        "precip",
        "t2m",
        "pev",
        "E",
        "SMsurf",
        "SMroot",
        "Eb",
        "sp",
        "tp",
        "ndvi",
        "p84.162",
        "boku_VCI",
        "VCI3M",
        "modis_ndvi",
    ]
    target_vars = ["boku_VCI"]
    dynamic_vars = ["precip", "t2m", "pet", "E", "SMroot", "SMsurf", "VCI3M"]
    static = True

    vars_to_include = target_vars + dynamic_vars
    vars_to_exclude = [v for v in all_vars if v not in vars_to_include]

    data_dir = get_data_path()
    # data_dir = Path('/Volumes/Lees_Extend/data/ecmwf_sowc/data')

    if (data_dir / "features/one_month_forecast").exists():
        from_path = data_dir / "features/one_month_forecast"
        to_path = data_dir / "features/__one_month_forecast"
        _rename_directory(from_path, to_path, with_datetime=True)

    target_var = target_vars[0]
    run_experiments(
        vars_to_exclude=vars_to_exclude, data_dir=data_dir, target_var=target_var
    )

    move_all_experiment_files(
        data_dir=data_dir, folder_name=f"robustness_{target_var}", dir_="models"
    )

    move_all_experiment_files(
        data_dir=data_dir, folder_name=f"robustness_{target_var}", dir_="features"
    )
