from pathlib import Path
from typing import List
import sys

sys.path.append("../..")

from _base_models import parsimonious, regression, linear_nn, rnn, earnn

from scripts.utils import _rename_directory, get_data_path
from src.engineer import Engineer


def engineer(
    experiment="one_month_forecast",
    process_static=False,
    pred_months=3,
    target_var="boku_VCI",
):
    engineer = Engineer(get_data_path(), experiment=experiment, process_static=False)
    engineer.engineer(
        test_year=[y for y in range(2016, 2019)],
        target_variable=target_var,
        pred_months=pred_months,
        expected_length=pred_months,
    )


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


def rename_model_experiment_file(
    data_dir: Path, vars_: List[str], static: bool, target_var: str
) -> None:
    vars_joined = "_".join(vars_)
    from_path = data_dir / "models" / "one_month_forecast"
    if static:
        to_path = (
            data_dir
            / "models"
            / f"one_month_forecast_{vars_joined}_YESstatic_{target_var}"
        )
    else:
        to_path = (
            data_dir
            / "models"
            / f"one_month_forecast_{vars_joined}_NOstatic_{target_var}"
        )

    # with_datetime ensures that unique
    _rename_directory(from_path, to_path, with_datetime=True)


def main(target_var, all_vars):
    # RUN engineer
    engineer(target_var=target_var)

    autoregressive = [target_var]  # 'VCI3M'
    dynamic = ["precip", "t2m", "pet", "E", "SMroot", "SMsurf"]
    static_list = [False, False, True]

    for vars_to_include, static_bool in zip(
        [autoregressive, autoregressive + dynamic, autoregressive + dynamic],
        static_list,
    ):
        print(
            f'\n{"-" * 10}\nRunning experiment with: {vars_to_include} with static: {static_bool} for {target_var}\n{"-" * 10}'
        )

        # FIT models
        vars_to_exclude = [v for v in all_vars if v not in vars_to_include]

        parsimonious()
        if static_bool:
            lstm(vars_to_exclude, static="features")
            ealstm(vars_to_exclude, static="features")
        else:
            lstm(vars_to_exclude, static=None)

        # RENAME model directories
        data_dir = get_data_path()
        rename_model_experiment_file(
            data_dir, vars_to_include, static=static_bool, target_var=target_var
        )


if __name__ == "__main__":
    # run the autoregressive LSTM
    # run the autoregressive + dynamic LSTM
    # run the autoregressive + dynamic + static LSTM / EALSTM
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

    target_vars = ["boku_VCI"]  # , "VCI"]

    # run the engineer THEN models
    for target_var in target_vars:
        main(target_var, all_vars)
