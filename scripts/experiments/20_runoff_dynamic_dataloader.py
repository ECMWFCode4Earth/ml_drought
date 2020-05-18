"""
# from drought TO RUNOFF
mv interim interim_; mv features features_; mv features__ features; mv interim__ interim

# from runoff TO DROUGHT
mv features features__; mv interim interim__; mv interim_ interim ; mv features_ features
"""

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
from scripts.utils import (
    _rename_directory,
    get_data_path,
    rename_features_dir,
    rename_models_dir,
)
from scripts.experiments._static_ignore_vars import static_ignore_vars

from src.engineer.dynamic_engineer import DynamicEngineer
from src.models import EARecurrentNetwork
from src.models import load_model


def engineer(
    data_dir,
    static_ignore_vars,
    dynamic_ignore_vars,
    logy=True,
    test_years=np.arange(2011, 2017),
    stations=None,
) -> None:
    de = DynamicEngineer(data_dir, process_static=True)

    de.engineer(
        augment_static=False,
        static_ignore_vars=static_ignore_vars,
        dynamic_ignore_vars=dynamic_ignore_vars,
        logy=logy,
        test_years=test_years,
        spatial_units=stations,
    )
    print("\n\n** Data Engineered! **\n\n")


def train_model(
    data_dir,
    static_ignore_vars,
    dynamic_ignore_vars,
    n_epochs=100,
    seq_length=365,
    test_years=np.arange(2011, 2017),
    target_var="discharge_spec",
    batch_size=1000,
    static_embedding_size=64,
    hidden_size=128,
    early_stopping: Optional[int] = None,
    dense_features: Optional[List[int]] = None,
    rnn_dropout: float = 0.25,
    dropout: float = 0.25,
    loss_func: str = "MSE",
    forecast_horizon: int = 1,
    normalize_y: bool = True,
    learning_rate: float = 1e-3,
    clip_zeros: bool = False,
) -> EARecurrentNetwork:
    # initialise the model
    ealstm = EARecurrentNetwork(
        data_folder=data_dir,
        batch_size=batch_size,
        hidden_size=hidden_size,
        experiment="one_timestep_forecast",
        dynamic=True,
        seq_length=seq_length,
        dynamic_ignore_vars=dynamic_ignore_vars,
        static_ignore_vars=static_ignore_vars,
        target_var=target_var,
        test_years=test_years,
        static_embedding_size=static_embedding_size,
        dense_features=dense_features,
        rnn_dropout=rnn_dropout,
        dropout=dropout,
        forecast_horizon=forecast_horizon,
        include_latlons=False,
        include_pred_month=False,
        include_timestep_aggs=False,
        include_yearly_aggs=False,
        normalize_y=True,
    )
    assert ealstm.seq_length == seq_length
    print("\n\n** Initialised Models! **\n\n")

    # Train the model on train set
    train_rmses, train_losses, val_rmses = ealstm.train(
        num_epochs=n_epochs,
        early_stopping=early_stopping,
        loss_func=loss_func,
        learning_rate=learning_rate,
        clip_zeros=clip_zeros,
    )
    print("\n\n** Model Trained! **\n\n")

    # save the model
    ealstm.save_model()
    pickle.dump(train_rmses, open(ealstm.model_dir / "train_rmses.pkl", "wb"))
    pickle.dump(train_losses, open(ealstm.model_dir / "train_losses.pkl", "wb"))
    pickle.dump(val_rmses, open(ealstm.model_dir / "val_rmses.pkl", "wb"))

    return ealstm


def run_evaluation(data_dir, ealstm=None):
    print("** Running Model Evaluation **")
    if ealstm is None:
        ealstm = load_model(
            data_dir / "models/one_timestep_forecast/ealstm/model.pt", device="cpu"
        )
    ealstm.batch_size = 10
    # move to CPU
    ealstm.move_model("cpu")

    # evaluate on the test set
    ealstm.evaluate(spatial_unit_name="station_id", save_preds=True)
    results_dict = json.load(
        open(data_dir / "models/one_timestep_forecast/ealstm/results.json", "rb")
    )
    print("** Overall RMSE: ", results_dict["total"], " **\n\n")


def main(
    engineer_only: bool = False,
    model_only: bool = False,
    reset_data_files: bool = False,
):
    data_dir = get_data_path()
    # ----------------------------------
    # Setup the experiment
    # ----------------------------------
    if reset_data_files:
        # check if features or models exists
        if (data_dir / "features").exists():
            rename_features_dir(data_dir)
        if (data_dir / "models").exists():
            rename_models_dir(data_dir)

    # ----------------------------------------------------------------
    # PARAMETERS
    # General Vars
    # dynamic_ignore_vars = ['discharge_vol', 'discharge_spec', 'pet']
    dynamic_ignore_vars = [
        # "temperature",
        "discharge_vol",
        "discharge_spec",
        "pet",
        # "humidity",
        # "shortwave_rad",
        # "longwave_rad",
        # "windspeed",
        # 'peti', 'precipitation',
    ]
    target_var = "discharge_spec"
    seq_length = 365 * 2
    forecast_horizon = 0
    logy = True
    batch_size = 300  # 1000 2000
    # catchment_ids = ["12002", "15006", "27009", "27034", "27041", "39001", "39081", "43021", "47001", "54001", "54057", "71001", "84013",]
    # catchment_ids = [int(c_id) for c_id in catchment_ids]
    catchment_ids = None

    # Model Vars
    num_epochs = 100  # 100
    test_years = np.arange(2014, 2016)
    static_embedding_size = 64  # 64
    hidden_size = 256  #  128
    # early_stopping = None
    early_stopping = 10
    dense_features = [128, 64]
    rnn_dropout = 0.3
    dropout = 0.3
    loss_func = "MSE"  # "MSE" "NSE"
    normalize_y = True
    learning_rate = 1e-4  # 5e-4
    clip_zeros = False

    if logy:
        assert (
            clip_zeros == False
        ), "Can only clip zero flows if training on raw discharge_spec"

    # ----------------------------------------------------------------
    # CODE
    if not model_only:
        engineer(
            data_dir=data_dir,
            static_ignore_vars=static_ignore_vars,
            dynamic_ignore_vars=dynamic_ignore_vars,
            logy=logy,
            test_years=test_years,
            stations=catchment_ids,
        )

    if not engineer_only:
        ealstm = train_model(
            data_dir=data_dir,
            static_ignore_vars=static_ignore_vars,
            dynamic_ignore_vars=dynamic_ignore_vars,
            n_epochs=num_epochs,
            seq_length=seq_length,
            test_years=test_years,
            target_var=target_var,
            batch_size=batch_size,
            static_embedding_size=static_embedding_size,
            hidden_size=hidden_size,
            early_stopping=early_stopping,
            dense_features=dense_features,
            rnn_dropout=rnn_dropout,
            loss_func=loss_func,
            dropout=dropout,
            forecast_horizon=forecast_horizon,
            normalize_y=normalize_y,
            learning_rate=learning_rate,
            clip_zeros=clip_zeros,
        )
        run_evaluation(data_dir, ealstm)

        # datestamp the model directory so that we can run multiple experiments
        # _rename_directory(
        #     from_path=data_dir / "models" / "one_timestep_forecast",
        #     to_path=data_dir / "models" / "one_timestep_forecast",
        #     with_datetime=True,
        # )


def evaluate_only():
    data_dir = get_data_path()
    run_evaluation(data_dir, ealstm=None)


if __name__ == "__main__":
    engineer_only = False
    model_only = False
    reset_data_files = False
    main(
        model_only=model_only,
        engineer_only=engineer_only,
        reset_data_files=reset_data_files,
    )
    # evaluate_only()
