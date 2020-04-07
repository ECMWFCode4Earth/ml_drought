from pathlib import Path
import xarray as xr
import numpy as np
from collections import defaultdict
import pickle
import pandas as pd
import json
import sys
sys.path.append("../..")

from typing import DefaultDict, Dict, Tuple, Optional, Union, List, Any
from scripts.utils import _rename_directory, get_data_path
from scripts.experiments._static_ignore_vars import static_ignore_vars

from src.engineer.dynamic_engineer import DynamicEngineer
from src.models import EARecurrentNetwork


def engineer(
    data_dir,
    static_ignore_vars,
    dynamic_ignore_vars,
    logy=True,
    test_years=np.arange(2011, 2017),
) -> None:
    de = DynamicEngineer(data_dir, process_static=True)

    de.engineer(
        augment_static=False,
        static_ignore_vars=static_ignore_vars,
        dynamic_ignore_vars=dynamic_ignore_vars,
        logy=logy,
        test_years=test_years,
    )
    print("\n\n** Data Engineered! **\n\n")


def run_model(
    data_dir,
    static_ignore_vars,
    dynamic_ignore_vars,
    n_epochs=100,
    seq_length=365,
    test_years=np.arange(2011, 2017),
    target_var = "discharge_spec",
) -> None:
    # initialise the model
    ealstm = EARecurrentNetwork(
        data_folder=data_dir,
        batch_size=2000,
        hidden_size=128,
        experiment='one_timestep_forecast',
        dynamic=True,
        seq_length=seq_length,
        dynamic_ignore_vars=dynamic_ignore_vars,
        static_ignore_vars=static_ignore_vars,
        target_var=target_var,
        test_years=np.arange(2011, 2017),
    )
    print("\n\n** Initialised Models! **\n\n")

    # Train the model on train set
    ealstm.train(num_epochs=n_epochs)
    print("\n\n** Model Trained! **\n\n")

    # evaluate on the test set
    ealstm.evaluate(
        spatial_unit_name='station_id',
        save_preds=True
    )
    results_dict = json.load(open(data_dir / 'models/one_timestep_forecast/ealstm/results.json', 'rb'))
    print("** Overall RMSE: ", results_dict['total'], " **\n\n")

    # save the model
    ealstm.save_model()

def main():
    data_dir = get_data_path()

    # ----------------------------------------------------------------
    # PARAMETERS
    # General Vars
    dynamic_ignore_vars = ['discharge_vol', 'discharge_spec']
    # dynamic_ignore_vars = ['temperature', 'discharge_vol', 'discharge_spec',
    #            'pet', 'humidity', 'shortwave_rad', 'longwave_rad', 'windspeed']
    target_var = "discharge_spec"
    seq_length = 365
    forecast_horizon = 1
    logy = True

    # Model Vars
    test_years = [2011, 2012, 2013, 2014, 2015]
    num_epochs = 10

    # ----------------------------------------------------------------
    # CODE
    engineer(
        data_dir=data_dir,
        static_ignore_vars=static_ignore_vars,
        dynamic_ignore_vars=dynamic_ignore_vars,
        logy=logy,
        test_years=test_years
    )

    run_model(
        data_dir=data_dir,
        static_ignore_vars=static_ignore_vars,
        dynamic_ignore_vars=dynamic_ignore_vars,
        n_epochs=num_epochs,
        seq_length=seq_length,
        test_years=test_years,
        target_var=target_var,
    )

    # datestamp the model directory so that we can run multiple experiments
    _rename_directory(
        from_path=data_dir / "models" / "one_timestep_forecast",
        to_path=data_dir / "models" / "one_timestep_forecast",
        with_datetime=True,
    )


if __name__ == "__main__":
    main()