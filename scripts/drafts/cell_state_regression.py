from pathlib import Path
import pandas as pd

import sys
sys.path.insert(2, "/home/tommy/ml_drought")

from scripts.drafts.gb_sm_data import read_gb_sm_data
from scripts.drafts.cell_state_extract import (load_normalised_cs_data,
                                               normalize_xr_by_basin,
                                               load_ealstm,
                                               load_config_file,
                                               load_lstm,)

# CellStateDataset
# Dataloader
# train-test split
# initialise model
# train model on each soil level
# test models on each soil level

# Dataset

# Train Test Split

# ALL Training Process




if __name__ == "__main__":
    EALSTM: bool = False
    data_dir = Path("/cats/datastore/data/")
    assert data_dir.exists()

    catchment_ids = [
        int(c)
        for c in [
            "12002",
            "15006",
            "27009",
            "27034",
            "27041",
            "39001",
            "39081",
            "43021",
            "47001",
            "54001",
            "54057",
            "71001",
            "84013",
        ]
    ]

    if EALSTM:
        run_dir = data_dir / "runs/ensemble_EALSTM/ealstm_ensemble6_nse_1998_2008_2910_030601"
        config = load_config_file(run_dir)
        model = load_ealstm(config)
    else:
        run_dir = data_dir / "runs/ensemble/lstm_ensemble6_nse_1998_2008_2710_171032"
        config = load_config_file(run_dir)
        model = load_lstm(config)

    TEST_BASINS = [str(id_) for id_ in catchment_ids]
    FINAL_VALUE = True
    TEST_TIMES = pd.date_range(
        config.test_start_date, config.test_end_date, freq="D"
    )

    # get the normalised input data
    norm_cs_data = load_normalised_cs_data(
        config=config,
        model=model,
        test_basins=TEST_BASINS,
        test_times=TEST_TIMES,
        final_value=FINAL_VALUE,
    )

    # get the target SM data
    sm = read_gb_sm_data(data_dir)
    norm_sm = normalize_xr_by_basin(sm)

    # create input/target data
    if not FINAL_VALUE:
        # need to collapse seq_length by taking a mean over
        #  repeated "actual time" values
        input_data = norm_cs_data.groupby("time").mean()
    else:
        input_data = norm_cs_data

    target_data = norm_sm

    # CellStateDataset
    # Dataloader
    # train-test split
    # initialise model
    # train model on each soil level
    # test models on each soil level
