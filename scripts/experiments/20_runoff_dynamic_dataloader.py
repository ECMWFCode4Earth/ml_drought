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
from src.models import EARecurrentNetwork, RecurrentNetwork
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


def train_lstm(
    data_dir,
    static_ignore_vars,
    dynamic_ignore_vars,
    n_epochs=100,
    seq_length=365,
    test_years=np.arange(2011, 2017),
    target_var="discharge_spec",
    batch_size=1000,
    static_embedding_size=None,
    hidden_size=128,
    early_stopping: Optional[int] = None,
    dense_features: Optional[List[int]] = None,
    rnn_dropout: float = 0.25,
    dropout: float = 0.25,
    loss_func: str = "MSE",
    forecast_horizon: int = 1,
    normalize_y: bool = True,
    learning_rate: float = 1e-3,
    static: Optional[str] = "features",
    clip_values_to_zero: bool = False,
    train_years: Optional[List[int]] = None,
    val_years: Optional[List[int]] = None,
    batch_file_size: int = 3,
) -> RecurrentNetwork:
    lstm = RecurrentNetwork(  # type: ignore
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
        dense_features=dense_features,
        rnn_dropout=rnn_dropout,
        dropout=dropout,
        forecast_horizon=forecast_horizon,
        include_latlons=False,
        include_pred_month=False,
        include_timestep_aggs=False,
        include_yearly_aggs=False,
        normalize_y=True,
        static=static,
        clip_values_to_zero=clip_values_to_zero,
        train_years=train_years,
        val_years=val_years,
    )

    # Train the model on train set
    print("\n** Training LSTM **\n")
    train_rmses, train_losses, val_rmses = lstm.train(
        num_epochs=n_epochs,
        early_stopping=early_stopping,
        loss_func=loss_func,
        learning_rate=learning_rate,
        batch_file_size=batch_file_size,
        # clip_zeros=clip_zeros,
    )
    print("\n\n** LSTM Model Trained! **\n\n")

    # save the model
    lstm.save_model()
    pickle.dump(train_rmses, open(lstm.model_dir / "train_rmses.pkl", "wb"))
    pickle.dump(train_losses, open(lstm.model_dir / "train_losses.pkl", "wb"))
    pickle.dump(val_rmses, open(lstm.model_dir / "val_rmses.pkl", "wb"))

    return lstm


def train_ealstm(
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
    static: Optional[str] = "features",
    clip_values_to_zero: bool = False,
    train_years: Optional[List[int]] = None,
    val_years: Optional[List[int]] = None,
    batch_file_size: int = 3,
) -> EARecurrentNetwork:
    # initialise the model
    ealstm = EARecurrentNetwork(  # type: ignore
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
        static=static,
        clip_values_to_zero=clip_values_to_zero,
        train_years=train_years,
        val_years=val_years,
    )
    assert ealstm.seq_length == seq_length
    print("\n\n** Training EALSTM! **\n\n")

    # Train the model on train set
    train_rmses, train_losses, val_rmses = ealstm.train(
        num_epochs=n_epochs,
        early_stopping=early_stopping,
        loss_func=loss_func,
        learning_rate=learning_rate,
        batch_file_size=batch_file_size,
        # clip_zeros=clip_zeros,
    )
    print("\n\n** EALSTM Model Trained! **\n\n")

    # save the model
    ealstm.save_model()
    pickle.dump(train_rmses, open(ealstm.model_dir / "train_rmses.pkl", "wb"))
    pickle.dump(train_losses, open(ealstm.model_dir / "train_losses.pkl", "wb"))
    pickle.dump(val_rmses, open(ealstm.model_dir / "val_rmses.pkl", "wb"))

    return ealstm


def run_evaluation(data_dir, model=None, batch_file_size: int = 3):
    print("** Running Model Evaluation **")
    if model is None:  # load the ealstm by default
        model = load_model(
            data_dir / "models/one_timestep_forecast/ealstm/model.pt", device="cpu"
        )
    model.batch_size = 10
    # move to CPU
    model.move_model("cpu")

    # evaluate on the test set
    model.evaluate(
        spatial_unit_name="station_id", save_preds=True, batch_file_size=batch_file_size
    )
    # results_dict = json.load(
    #     open(data_dir / f"models/one_timestep_forecast/{model.name}/results.json", "rb")
    # )
    # print("** Overall RMSE: ", results_dict["total"], " **\n\n")


def main(
    engineer_only: bool = False,
    model_only: bool = False,
    reset_data_files: bool = False,
):
    data_dir = get_data_path()
    # data_dir = Path("/Volumes/Lees_Extend/data/ecmwf_sowc/data")
    assert data_dir.exists()

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
        "temperature",
        "discharge_vol",
        "discharge_spec",
        "pet",
        "humidity",
        "shortwave_rad",
        "longwave_rad",
        "windspeed",
        # 'peti', 'precipitation',
    ]
    target_var = "discharge_spec"  # discharge_spec  discharge_vol
    seq_length = 365  # * 2
    forecast_horizon = 0
    logy = True
    batch_size = 50  # 1000 2000
    batch_file_size = 5
    # catchment_ids = ["12002", "15006", "27009", "27034", "27041", "39001", "39081", "43021", "47001", "54001", "54057", "71001", "84013",]
    # catchment_ids = [int(c_id) for c_id in catchment_ids]
    catchment_ids = None

    # Model Vars
    num_epochs = 100
    test_years = np.arange(2010, 2016)
    static_embedding_size = 64  # 64
    hidden_size = 256  #  128
    # early_stopping = None
    early_stopping = 5
    dense_features: List[int] = []  # [128, 64]
    rnn_dropout = 0
    dropout = 0
    loss_func = "huber"  # "MSE" "NSE" "huber"
    normalize_y = True
    learning_rate = {0: 1e-3, 5: 5e-4, 11: 1e-4}  # 1e-4  # 5e-4
    clip_values_to_zero = False
    static = "features"  #  embedding features None

    train_years = np.arange(1995, 2010)  #  np.arange(1979, 2010)
    val_years = np.arange(1990, 1992)

    if logy:
        assert clip_values_to_zero is False, "Don't clip to zero if log transform y"

    assert not any(np.isin(test_years, train_years)), "MODEL LEAKAGE"
    assert not any(np.isin(test_years, val_years)), "MODEL LEAKAGE"

    # if running on Tommy's machine (DEBUG)
    try:
        if data_dir.parents[3].name == ("Volumes"):
            model_only = True
            num_epochs = 1
    except IndexError:
        pass

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

    model_kwargs = dict(
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
        static=static,
        clip_values_to_zero=clip_values_to_zero,
        train_years=train_years,
        val_years=val_years,
        batch_file_size=batch_file_size,
    )
    if not engineer_only:
        lstm = train_lstm(**model_kwargs)
        run_evaluation(data_dir, lstm)

        # ealstm = train_ealstm(**model_kwargs)
        # run_evaluation(data_dir, ealstm)

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