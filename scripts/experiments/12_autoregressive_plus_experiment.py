from pathlib import Path
from typing import List
import sys

sys.path.append("../..")

from _base_models import parsimonious, regression, linear_nn, rnn, earnn

from scripts.utils import _rename_directory, get_data_path


def lstm(ignore_vars, static):
    rnn(
        experiment="one_month_forecast",
        surrounding_pixels=None,
        explain=False,
        ignore_vars=ignore_vars,
        num_epochs=50,
        early_stopping=5,
        hidden_size=256,
        # static data
        static=static,
        include_pred_month=True,  # if static is not None else False,
        include_latlons=True if static is not None else False,
        include_prev_y=True,  # if static is not None else False,
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
        early_stopping=5,
        hidden_size=256,
        static_embedding_size=64,
        # static data
        static=static,
        include_pred_month=True,
        include_latlons=True,
        include_prev_y=True,
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
        "boku_vci",
        "VCI3M",
        "modis_ndvi",
    ]

    autoregressive = ["boku_vci"]  # 'VCI3M'
    dynamic = ["precip", "t2m", "pet", "E", "SMroot", "SMsurf"]
    static_list = [False, False, True]

    for vars_to_include, static_bool in zip(
        [autoregressive, autoregressive + dynamic, autoregressive + dynamic],
        static_list,
    ):
        print(
            f'\n{"-" * 10}\nRunning experiment with: {vars_to_include} with static: {static_bool}\n{"-" * 10}'
        )

        vars_to_exclude = [v for v in all_vars if v not in vars_to_include]
        if static_bool:
            lstm(vars_to_exclude, static="features")
            ealstm(vars_to_exclude, static="features")
        else:
            lstm(vars_to_exclude, static=None)
