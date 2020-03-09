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
    earnn(
        experiment="one_timestep_forecast",
        include_pred_month=True,
        surrounding_pixels=None,
        pretrained=False,
        explain=False,
        static="features",
        ignore_vars=None,
        num_epochs=50,
        early_stopping=5,
        hidden_size=256,
        static_embedding_size=64,
        include_latlons=False,
        # yearly_aggs=False,
    )


if __name__ == "__main__":
    # engineer(seq_length=3)
    # engineer_static_only()
    run_model()