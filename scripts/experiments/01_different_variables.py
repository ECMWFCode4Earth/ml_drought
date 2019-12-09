"""
run_different_variables.py

All vars:
['pev', 'sp', 't2m', 'tp', 'VCI', 'precip', 'ndvi', 'E', 'Eb', 'SMroot', 'SMsurf',]
"""
from itertools import combinations
from pathlib import Path
from typing import List
import sys

sys.path.append("../..")

from src.models import (
    Persistence,
    LinearRegression,
    LinearNetwork,
    RecurrentNetwork,
    EARecurrentNetwork,
    load_model,
)
from scripts.utils import _rename_directory


# NOTE: p84.162 == 'vertical integral of moisture flux'


def rename_model_experiment_file(vars_: List[str], static: bool) -> None:
    if Path(".").absolute().as_posix().split("/")[-1] == "ml_drought":
        data_dir = Path("data")
    else:
        data_dir = Path("../data")

    vars_joined = "_".join(vars_)
    from_path = data_dir / "models" / "one_month_forecast"
    if static:
        to_path = data_dir / "models" / f"one_month_forecast_{vars_joined}_YESstatic"
    else:
        to_path = data_dir / "models" / f"one_month_forecast_{vars_joined}_NOstatic"

    _rename_directory(from_path, to_path)


def run_all_models_as_experiments(
    vars_to_include: List[str],
    ignore_vars: List[str],
    static: bool,
    run_regression: bool = True,
):
    print(f"Experiment {vars_to_include} Static: {static}")

    # RUN EXPERIMENTS
    if run_regression:
        regression(ignore_vars=ignore_vars, include_static=static)

    linear_nn(ignore_vars=ignore_vars, include_static=static)
    rnn(ignore_vars=ignore_vars, include_static=static)
    earnn(pretrained=False, ignore_vars=ignore_vars, include_static=static)

    # RENAME DIRECTORY
    rename_model_experiment_file(vars_to_include, static)
    print(f"Experiment {vars_to_include} finished")


def parsimonious(experiment="one_month_forecast",):
    # if the working directory is alread ml_drought don't need ../data
    if Path(".").absolute().as_posix().split("/")[-1] == "ml_drought":
        data_path = Path("data")
    else:
        data_path = Path("../data")

    predictor = Persistence(data_path, experiment=experiment)
    predictor.evaluate(save_preds=True)


def regression(
    experiment="one_month_forecast",
    include_pred_month=True,
    surrounding_pixels=None,
    ignore_vars=None,
    include_static=True,
):
    # if the working directory is alread ml_drought don't need ../data
    if Path(".").absolute().as_posix().split("/")[-1] == "ml_drought":
        data_path = Path("data")
    else:
        data_path = Path("../data")

    predictor = LinearRegression(
        data_path,
        experiment=experiment,
        include_pred_month=include_pred_month,
        surrounding_pixels=surrounding_pixels,
        ignore_vars=ignore_vars,
        include_static=include_static,
    )
    predictor.train(early_stopping=5)
    predictor.evaluate(save_preds=True)

    # mostly to test it works
    # predictor.explain(save_shap_values=True)


def linear_nn(
    experiment="one_month_forecast",
    include_pred_month=True,
    surrounding_pixels=None,
    ignore_vars=None,
    include_static=True,
):
    # if the working directory is alread ml_drought don't need ../data
    if Path(".").absolute().as_posix().split("/")[-1] == "ml_drought":
        data_path = Path("data")
    else:
        data_path = Path("../data")

    predictor = LinearNetwork(
        layer_sizes=[100],
        data_folder=data_path,
        experiment=experiment,
        include_pred_month=include_pred_month,
        surrounding_pixels=surrounding_pixels,
        ignore_vars=ignore_vars,
        include_static=include_static,
    )
    predictor.train(num_epochs=50, early_stopping=5)
    predictor.evaluate(save_preds=True)
    predictor.save_model()

    # _ = predictor.explain(save_shap_values=True)


def rnn(
    experiment="one_month_forecast",
    include_pred_month=True,
    surrounding_pixels=None,
    ignore_vars=None,
    include_static=True,
):
    # if the working directory is alread ml_drought don't need ../data
    if Path(".").absolute().as_posix().split("/")[-1] == "ml_drought":
        data_path = Path("data")
    else:
        data_path = Path("../data")

    predictor = RecurrentNetwork(
        hidden_size=128,
        data_folder=data_path,
        experiment=experiment,
        include_pred_month=include_pred_month,
        surrounding_pixels=surrounding_pixels,
        ignore_vars=ignore_vars,
        include_static=include_static,
    )
    predictor.train(num_epochs=50, early_stopping=5)
    predictor.evaluate(save_preds=True)
    predictor.save_model()

    # _ = predictor.explain(save_shap_values=True)


def earnn(
    experiment="one_month_forecast",
    include_pred_month=True,
    surrounding_pixels=None,
    pretrained=True,
    ignore_vars=None,
    include_static=True,
):
    # if the working directory is alread ml_drought don't need ../data
    if Path(".").absolute().as_posix().split("/")[-1] == "ml_drought":
        data_path = Path("data")
    else:
        data_path = Path("../data")

    if not pretrained:
        predictor = EARecurrentNetwork(
            hidden_size=128,
            data_folder=data_path,
            experiment=experiment,
            include_pred_month=include_pred_month,
            surrounding_pixels=surrounding_pixels,
            ignore_vars=ignore_vars,
            include_static=include_static,
        )
        predictor.train(num_epochs=50, early_stopping=10)
        predictor.evaluate(save_preds=True)
        predictor.save_model()
    else:
        predictor = load_model(data_path / f"models/{experiment}/ealstm/model.pt")

    # test_file = data_path / f'features/{experiment}/test/2018_3'
    # assert test_file.exists()
    # all_shap_for_file(test_file, predictor, batch_size=100)


if __name__ == "__main__":
    ignore_vars = None
    always_ignore_vars = ["ndvi", "p84.162", "sp", "tp", "Eb"]
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
    ]
    important_vars = ["VCI", "precip", "t2m", "pev", "E", "SMsurf", "SMroot"]
    additional_vars = ["precip", "t2m", "pev", "E", "SMsurf", "SMroot"]

    # create ALL combinations of features (VCI + ...)
    expts_to_run = [["VCI"]]
    for n_combinations in range(len(additional_vars)):
        expts_to_run.extend(
            [["VCI"] + list(c) for c in combinations(additional_vars, n_combinations)]
        )

    # # add variables in one at a time
    # for i in range(len(important_vars)):
    #     vars_to_include = important_vars[:-(i)]
    #     if vars_to_include == []:
    #         continue

    for vars_to_include in expts_to_run:
        print(f'\n{"-" * 10}\nRunning experiment with: {vars_to_include}\n{"-" * 10}')

        vars_to_exclude = [v for v in important_vars if v not in vars_to_include]
        ignore_vars = always_ignore_vars + vars_to_exclude
        print(ignore_vars)

        # run experiments
        for static in [True, False]:
            run_all_models_as_experiments(
                vars_to_include, ignore_vars, static=static, run_regression=False,
            )
