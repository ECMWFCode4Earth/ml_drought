"""
run_different_variables.py

All vars:
['pev', 'sp', 't2m', 'tp', 'VCI', 'precip', 'ndvi', 'E', 'Eb', 'SMroot', 'SMsurf',]

# NOTE: p84.162 == 'vertical integral of moisture flux'

"""
from itertools import combinations
from pathlib import Path
from typing import List
import sys

sys.path.append("../..")

from _base_models import parsimonious, regression, linear_nn, rnn, earnn

from scripts.utils import _rename_directory, get_data_path


def rename_model_experiment_file(
    data_dir: Path, vars_: List[str], static: bool
) -> None:
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

    if static:
        # 'embeddings' or 'features'
        linear_nn(ignore_vars=ignore_vars, static="embeddings")
        rnn(ignore_vars=ignore_vars, static="embeddings")
        earnn(pretrained=False, ignore_vars=ignore_vars, static="embeddings")
    else:
        linear_nn(ignore_vars=ignore_vars, static=None)
        rnn(ignore_vars=ignore_vars, static=None)
        earnn(pretrained=False, ignore_vars=ignore_vars, static=None)

    # RENAME DIRECTORY
    data_dir = get_data_path()
    rename_model_experiment_file(data_dir, vars_to_include, static)
    print(f"Experiment {vars_to_include} finished")


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
    expts_to_run = []
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
                vars_to_include, ignore_vars, static=static, run_regression=False
            )
