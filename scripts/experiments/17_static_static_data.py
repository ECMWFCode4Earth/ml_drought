"""
# from drought TO RUNOFF
mv interim interim_; mv features features_; mv features__ features; mv interim__ interim

# from runoff TO DROUGHT
mv features features__; mv interim interim__; mv interim_ interim ; mv features_ features

# Experiment #7 is VCI3M results
# Experiment #9 is boku_VCI results

mv features 5_features; mv models 5_models;
mv 9_features features; mv 9_models models;
"""

import sys

sys.path.append("../..")

from scripts.utils import _rename_directory, get_data_path
from _base_models import regression, linear_nn, rnn, earnn, persistence, climatology
from src.engineer import Engineer
from pathlib import Path
from typing import Optional, List


def rename_features_dir(data_path: Path):
    """increment the features dir by 1"""
    old_paths = [d for d in data_path.glob("*_features*")]
    if old_paths == []:
        integer = 0
    else:
        old_max = max([int(p.name.split("_")[0]) for p in old_paths])
        integer = old_max + 1

    _rename_directory(
        from_path=data_path / "features",
        to_path=data_path / f"{integer}_features",
        with_datetime=False,
    )


def rename_models_dir(data_path: Path):
    old_paths = [d for d in data_path.glob("*_models*")]
    if old_paths == []:
        integer = 0
    else:
        old_max = max([int(p.name.split("_")[0]) for p in old_paths])
        integer = old_max + 1

    _rename_directory(
        from_path=data_path / "models",
        to_path=data_path / f"{integer}_models",
        with_datetime=False,
    )


def engineer(
    pred_months=3,
    target_var="boku_VCI",
    process_static=False,
    global_means: bool = True,
    log_vars: Optional[List[str]] = None,
):
    engineer = Engineer(
        get_data_path(), experiment="one_month_forecast", process_static=process_static
    )
    engineer.engineer(
        test_year=[y for y in range(2016, 2019)],
        target_variable=target_var,
        pred_months=pred_months,
        expected_length=pred_months,
        global_means=global_means,
    )


if __name__ == "__main__":
    data_path = get_data_path()

    # ----------------------------------
    # Setup the experiment
    # ----------------------------------
    # check if features or models exists
    if (data_path / "features").exists():
        rename_features_dir(data_path)
    if (data_path / "models").exists():
        rename_models_dir(data_path)

    # ----------------------------------
    # Run the Experiment
    # ----------------------------------
    # 1. Run the engineer
    target_var = "boku_VCI"  #  "VCI3M" "boku_VCI"
    pred_months = 3
    engineer(
        pred_months=pred_months,
        target_var=target_var,
        process_static=True,
        global_means=True,
    )

    # NOTE: why have we downloaded 2 variables for ERA5 evaporaton
    # important_vars = ["VCI", "precip", "t2m", "pev", "p0005", "SMsurf", "SMroot"]
    # always_ignore_vars = ["ndvi", "p84.162", "sp", "tp", "Eb", "E", "p0001"]
    important_vars = ["boku_VCI", "precip", "t2m", "pev", "E", "SMsurf"]

    # NOTE: if commented out then INCLUDED in the model
    always_ignore_vars = [
        "VCI",
        "p84.162",
        "sp",
        "tp",
        "Eb",
        "VCI1M",
        "RFE1M",
        "VCI3M",
        # "boku_VCI",
        "modis_ndvi",
        "SMroot",
        "lc_class",  # remove for good clustering (?)
        "lc_class_group",  # remove for good clustering (?)
        "slt",  #  remove for good clustering (?)
        "no_data_one_hot",
        "lichens_and_mosses_one_hot",
        "permanent_snow_and_ice_one_hot",
        "urban_areas_one_hot",
        "water_bodies_one_hot",
        "t2m",
        "SMsurf",
        # "pev",
        # "e",
        "E",
    ]

    assert target_var not in always_ignore_vars
    other_target = "boku_VCI" if target_var == "VCI3M" else "VCI3M"
    assert other_target in always_ignore_vars

    # -------------
    # Model Parameters
    # -------------
    num_epochs = 50
    early_stopping = 10
    hidden_size = 256
    static_size = 64
    # normalize_y = True

    # -------------
    # baseline models
    # -------------
    persistence()
    climatology()

    regression(
        ignore_vars=always_ignore_vars,
        experiment="one_month_forecast",
        include_pred_month=True,
        surrounding_pixels=None,
        explain=False,
    )

    # # gbdt(ignore_vars=always_ignore_vars)
    linear_nn(
        ignore_vars=always_ignore_vars,
        experiment="one_month_forecast",
        include_pred_month=True,
        surrounding_pixels=None,
        explain=False,
        num_epochs=num_epochs,
        early_stopping=early_stopping,
        layer_sizes=[hidden_size],
        include_latlons=True,
        include_yearly_aggs=False,
        clear_nans=True,
    )

    # -------------
    # LSTM
    # -------------
    rnn(
        experiment="one_month_forecast",
        include_pred_month=True,
        surrounding_pixels=None,
        explain=False,
        static="features",
        ignore_vars=always_ignore_vars,
        num_epochs=num_epochs,
        early_stopping=early_stopping,
        hidden_size=hidden_size,
        include_latlons=True,
        include_yearly_aggs=False,
        clear_nans=True,
        weight_observations=False,
    )

    # -------------
    # EALSTM
    # -------------
    earnn(
        experiment="one_month_forecast",
        include_pred_month=True,
        surrounding_pixels=None,
        pretrained=False,
        explain=False,
        static="features",
        ignore_vars=always_ignore_vars,
        num_epochs=num_epochs,
        early_stopping=early_stopping,
        hidden_size=hidden_size,
        static_embedding_size=static_size,
        include_latlons=True,
        include_yearly_aggs=False,
        clear_nans=True,
        weight_observations=False,
        pred_month_static=False,
    )

    # rename the output file
    data_path = get_data_path()

    # _rename_directory(
    #     from_path=data_path / "models" / "one_month_forecast",
    #     to_path=data_path / "models" / "one_month_forecast_BASE_static_vars",
    #     with_datetime=True,
    # )
