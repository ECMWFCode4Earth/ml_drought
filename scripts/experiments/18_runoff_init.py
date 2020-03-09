from pathlib import Path
from typing import List, Tuple, Dict
import sys
import itertools

sys.path.append("../..")

from _base_models import parsimonious, regression, linear_nn, rnn, earnn

from scripts.utils import _rename_directory, get_data_path
from src.engineer.one_timestep_forecast import _OneTimestepForecastEngineer


def engineer(seq_length=3, target_var="discharge_vol"):
    engineer = _OneTimestepForecastEngineer(get_data_path(), process_static=True)
    engineer.engineer(
        test_year=[y for y in range(2011, 2016)],
        target_variable=target_var,
        seq_length=seq_length,
        expected_length=seq_length,
    )


def engineer_static_only():
    engineer = _OneTimestepForecastEngineer(get_data_path(), process_static=True)
    engineer._process_static()


def run_model():
    ignore_vars = None
    ignore_vars = [
        "temperature",
        "discharge_spec",
        "peti",
        "humidity",
        "shortwave_rad",
        "longwave_rad",
        "windspeed",
    ]
    rnn(
        experiment="one_timestep_forecast",
        batch_size=100,
        include_pred_month=True,
        surrounding_pixels=None,
        explain=False,
        static=None,  # "features",
        ignore_vars=ignore_vars,
        num_epochs=1,  # 50,
        early_stopping=5,
        hidden_size=256,
        include_latlons=False,
        include_prev_y=False,
        pretrained=True,
        # yearly_aggs=False,
        # static_embedding_size=64,
    )


if __name__ == "__main__":
    # engineer(seq_length=3)
    # engineer_static_only()
    run_model()
