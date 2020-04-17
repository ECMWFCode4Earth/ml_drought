from pathlib import Path
from typing import List, Tuple, Dict
import sys
import itertools

sys.path.append("../..")

from src.analysis import all_explanations_for_file
from scripts.utils import _rename_directory, get_data_path

EXPERIMENT =      'one_month_forecast'
TRUE_EXPERIMENT = 'one_month_forecast'
TARGET_VAR =      'boku_VCI'

from src.models import load_model

data_dir = get_data_path()

ealstm = load_model(data_dir / 'models' / EXPERIMENT / 'ealstm' / 'model.pt')
ealstm.models_dir = data_dir / 'models' / EXPERIMENT

ealstm.experiment = TRUE_EXPERIMENT

all_explanations_for_file(data_dir / f'features/{EXPERIMENT}/test/2018_3', ealstm)