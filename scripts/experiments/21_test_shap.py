from pathlib import Path
from typing import List, Tuple, Dict
import sys
import itertools

sys.path.append("../..")

from src.analysis import all_explanations_for_file
from scripts.utils import _rename_directory, get_data_path

EXPERIMENT = "one_month_forecast"
TRUE_EXPERIMENT = "one_month_forecast"
TARGET_VAR = "boku_VCI"

from src.models import load_model

data_dir = get_data_path()

ealstm = load_model(data_dir / "models" / EXPERIMENT / "ealstm" / "model.pt")
ealstm.models_dir = data_dir / "models" / EXPERIMENT

ealstm.experiment = TRUE_EXPERIMENT

# all_explanations_for_file(data_dir / f'features/{EXPERIMENT}/test/2018_3', ealstm)

# get all training data
def get_test_dl(model):
    return model.get_dataloader(
        data_path=data_dir,
        mode="test",
        batch_file_size=1,
        to_tensor=True,
        shuffle_data=False,
    )


# initialise the parameters
loader = get_test_dl(ealstm)
date_string, test_data = list(next(iter(loader)).items())[0]
num_pixels = test_data.x.historical.shape[0]
start_idx = 0
var_names = test_data.x_vars

# for each test datetime in the test directory
loader = get_test_dl(ealstm)

test_data_explanations = {}
for key, val in loader:
    # run explanations
    explanations = ealstm.explain(
        x=val.x,
        var_names=var_names,
        background_size=100,  # how many values to take from train data
        start_idx=start_idx,
        num_inputs=num_pixels,
        method="shap",
    )
    test_data_explanations[key] = explanations


#
