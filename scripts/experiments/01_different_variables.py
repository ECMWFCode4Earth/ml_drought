"""
run_different_variables.py

All vars:
['pev', 'sp', 't2m', 'tp', 'VCI', 'precip', 'ndvi', 'E', 'Eb', 'SMroot', 'SMsurf',]

# NOTE: p84.162 == 'vertical integral of moisture flux'

Experiment ['VCI', 'E'] Static: False

Training rnn for experiment one_month_forecast
/home/cdsuser/miniconda3/envs/esowc-drought/lib/python3.7/site-packages/xarray/core/nanops.py:159: RuntimeWarning: Mean of empty slice
  return np.nanmean(a, axis=axis, dtype=dtype)
../../data/features/one_month_forecast/train/1985_3 returns no values. Skipping
/home/cdsuser/miniconda3/envs/esowc-drought/lib/python3.7/site-packages/xarray/core/nanops.py:159: RuntimeWarning: Mean of empty slice
  return np.nanmean(a, axis=axis, dtype=dtype)
../../data/features/one_month_forecast/train/2004_6 returns no values. Skipping
/home/cdsuser/miniconda3/envs/esowc-drought/lib/python3.7/site-packages/xarray/core/nanops.py:159: RuntimeWarning: Mean of empty slice
  return np.nanmean(a, axis=axis, dtype=dtype)
../../data/features/one_month_forecast/train/1995_3 returns no values. Skipping
/home/cdsuser/miniconda3/envs/esowc-drought/lib/python3.7/site-packages/xarray/core/nanops.py:159: RuntimeWarning: Mean of empty slice
  return np.nanmean(a, axis=axis, dtype=dtype)
../../data/features/one_month_forecast/train/1994_11 returns no values. Skipping
/home/cdsuser/miniconda3/envs/esowc-drought/lib/python3.7/site-packages/xarray/core/nanops.py:159: RuntimeWarning: Mean of empty slice
  return np.nanmean(a, axis=axis, dtype=dtype)
../../data/features/one_month_forecast/train/1994_12 returns no values. Skipping
../../data/features/one_month_forecast/train/1985_1 returns no values. Skipping
/home/cdsuser/miniconda3/envs/esowc-drought/lib/python3.7/site-packages/xarray/core/nanops.py:159: RuntimeWarning: Mean of empty slice
  return np.nanmean(a, axis=axis, dtype=dtype)
../../data/features/one_month_forecast/train/1995_2 returns no values. Skipping
/home/cdsuser/miniconda3/envs/esowc-drought/lib/python3.7/site-packages/xarray/core/nanops.py:159: RuntimeWarning: Mean of empty slice
  return np.nanmean(a, axis=axis, dtype=dtype)
../../data/features/one_month_forecast/train/1985_2 returns no values. Skipping
../../data/features/one_month_forecast/train/1994_10 returns no values. Skipping
/home/cdsuser/miniconda3/envs/esowc-drought/lib/python3.7/site-packages/xarray/core/nanops.py:159: RuntimeWarning: Mean of empty slice
  return np.nanmean(a, axis=axis, dtype=dtype)
../../data/features/one_month_forecast/train/1985_5 returns no values. Skipping
/home/cdsuser/miniconda3/envs/esowc-drought/lib/python3.7/site-packages/xarray/core/nanops.py:159: RuntimeWarning: Mean of empty slice
  return np.nanmean(a, axis=axis, dtype=dtype)
../../data/features/one_month_forecast/train/1985_4 returns no values. Skipping
../../data/features/one_month_forecast/train/2004_4 returns no values. Skipping
/home/cdsuser/miniconda3/envs/esowc-drought/lib/python3.7/site-packages/xarray/core/nanops.py:159: RuntimeWarning: Mean of empty slice
  return np.nanmean(a, axis=axis, dtype=dtype)
../../data/features/one_month_forecast/train/2004_8 returns no values. Skipping
/home/cdsuser/miniconda3/envs/esowc-drought/lib/python3.7/site-packages/xarray/core/nanops.py:159: RuntimeWarning: Mean of empty slice
  return np.nanmean(a, axis=axis, dtype=dtype)
../../data/features/one_month_forecast/train/1995_1 returns no values. Skipping
/home/cdsuser/miniconda3/envs/esowc-drought/lib/python3.7/site-packages/xarray/core/nanops.py:159: RuntimeWarning: Mean of empty slice
  return np.nanmean(a, axis=axis, dtype=dtype)
../../data/features/one_month_forecast/train/2004_5 returns no values. Skipping
/home/cdsuser/miniconda3/envs/esowc-drought/lib/python3.7/site-packages/xarray/core/nanops.py:159: RuntimeWarning: Mean of empty slice
  return np.nanmean(a, axis=axis, dtype=dtype)
../../data/features/one_month_forecast/train/2004_7 returns no values. Skipping
Traceback (most recent call last):
  File "01_different_variables.py", line 113, in <module>
    vars_to_include, ignore_vars, static=static, run_regression=False
  File "01_different_variables.py", line 54, in run_all_models_as_experiments
    rnn(ignore_vars=ignore_vars, static=None)
  File "/data/ml_drought/scripts/experiments/_base_models.py", line 94, in rnn
    predictor.train(num_epochs=num_epochs, early_stopping=early_stopping)
  File "../../src/models/neural_networks/base.py", line 125, in train
    *self._input_to_tuple(cast(Tuple[torch.Tensor, ...], x_batch))
  File "/home/cdsuser/miniconda3/envs/esowc-drought/lib/python3.7/site-packages/torch/nn/modules/module.py", line 493, in __call__
    result = self.forward(*input, **kwargs)
  File "../../src/models/neural_networks/rnn.py", line 276, in forward
    x = dense_layer(x)
  File "/home/cdsuser/miniconda3/envs/esowc-drought/lib/python3.7/site-packages/torch/nn/modules/module.py", line 493, in __call__
    result = self.forward(*input, **kwargs)
  File "/home/cdsuser/miniconda3/envs/esowc-drought/lib/python3.7/site-packages/torch/nn/modules/linear.py", line 92, in forward
    return F.linear(input, self.weight, self.bias)
  File "/home/cdsuser/miniconda3/envs/esowc-drought/lib/python3.7/site-packages/torch/nn/functional.py", line 1406, in linear
    ret = torch.addmm(bias, input, weight.t())
RuntimeError: invalid argument 13: ldc should be at least max(1, m=0), but have 0 at /opt/conda/conda-bld/pytorch-cpu_1556653114183/work/aten/src/TH/generic/THBlas.cpp:367
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

    # with_datetime ensures that unique
    _rename_directory(from_path, to_path, with_datetime=True)


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
        try:
            linear_nn(ignore_vars=ignore_vars, static="embeddings")
        except RuntimeError:
            print(
                f"\n{'*'*10}\n FAILED: LinearNN for vars={vars_to_include} static={static}\n{'*'*10}\n"
            )

        try:
            rnn(ignore_vars=ignore_vars, static="embeddings")
        except RuntimeError:
            print(
                f"\n{'*'*10}\n FAILED: RNN for vars={vars_to_include} static={static}\n{'*'*10}\n"
            )

        try:
            earnn(pretrained=False, ignore_vars=ignore_vars, static="embeddings")
        except RuntimeError:
            print(
                f"\n{'*'*10}\n FAILED: EALSTM for vars={vars_to_include} static={static}\n{'*'*10}\n"
            )

    else:
        try:
            linear_nn(ignore_vars=ignore_vars, static=None)
        except RuntimeError:
            print(
                f"\n{'*'*10}\n FAILED: LinearNN for vars={vars_to_include} static={static}\n{'*'*10}\n"
            )

        try:
            rnn(ignore_vars=ignore_vars, static=None)
        except RuntimeError:
            print(
                f"\n{'*'*10}\n FAILED: RNN for vars={vars_to_include} static={static}\n{'*'*10}\n"
            )

        try:
            earnn(pretrained=False, ignore_vars=ignore_vars, static=None)
        except RuntimeError:
            print(
                f"\n{'*'*10}\n FAILED: EALSTM for vars={vars_to_include} static={static}\n{'*'*10}\n"
            )

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

    for vars_to_include in expts_to_run[4:]:
        print(f'\n{"-" * 10}\nRunning experiment with: {vars_to_include}\n{"-" * 10}')

        vars_to_exclude = [v for v in important_vars if v not in vars_to_include]
        ignore_vars = always_ignore_vars + vars_to_exclude
        print(ignore_vars)

        # run experiments
        for static in [True, False]:
            try:
                run_all_models_as_experiments(
                    vars_to_include, ignore_vars, static=static, run_regression=False
                )
            except:
                print(
                    f'\n{"-" * 10}\Experiment FAILED: {vars_to_include} static:{static}\n{"-" * 10}'
                )
