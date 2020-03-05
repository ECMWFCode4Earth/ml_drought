from pathlib import Path
from typing import List, Tuple, Dict
import sys
import itertools

sys.path.append("../..")

from _base_models import parsimonious, regression, linear_nn, rnn, earnn

from scripts.utils import _rename_directory, get_data_path
from src.engineer.one_timestep_forecast import _OneTimestepForecastEngineer


def engineer(pred_months=3, target_var="discharge_vol"):
    engineer = _OneTimestepForecastEngineer(
        get_data_path(), process_static=True,
    )
    engineer.engineer(
        test_year=[y for y in range(2011, 2016)],
        target_variable=target_var,
        pred_months=pred_months,
        expected_length=pred_months,
    )


if __name__ == '__main__':
    engineer(pred_months=3)
