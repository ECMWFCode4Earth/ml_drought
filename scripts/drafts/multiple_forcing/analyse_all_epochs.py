"""
1. Run the training
    $ ipython --pdb main.py -- --config_file /path/to/config

2. Create Analysis objects (.nc, .csv)
    $ ipython --pdb analysis/analyse_all_epochs.py -- --run_dir /cats/datastore/data/runs/lstm_less_vars_2004_1507_1028
"""
import xarray as xr
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
from typing import Tuple, Dict, Optional, List

import sys

sys.path.append("/home/tommy/neuralhydrology")
from codebase.evaluation.tester import Tester
from codebase.config import read_config
from analysis.datautils import create_results_csv


def get_args() -> Dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str)
    args = vars(parser.parse_args())

    return args


def analyse_all_epochs(cfg: Dict, run_dir: Path) -> None:
    # TODO: only run the code if the Tester has not yet already been run for that epoch
    tester = Tester(cfg, run_dir)
    epochs: List[int] = [
        int(d.name.replace(".pt", "")[-3:]) for d in run_dir.glob("*.pt")
    ]
    all_dfs = []
    for epoch in epochs:
        # run evaluation (create the test_results.p object)
        tester.evaluate(epoch=epoch, save_results=True, metrics=cfg.get("metrics", []))
        # run analysis (create the .csv object)
        valid_df = create_results_csv(run_dir, epoch=epoch)
        if not all(np.isin(["station_id", "time"], [n for n in valid_df.index.names])):
            valid_df = valid_df.set_index(["time", "station_id"])

        assert all(np.isin(["station_id", "time"], [n for n in valid_df.index.names]))
        all_dfs.append(valid_df)

    # get the obs (ONCE)
    obs = valid_df["obs"]
    # rename the sim (for all)
    all_dfs = [
        df.rename(columns={"sim": f"sim_E{epoch:03}"}).drop(columns="obs")
        for epoch, df in zip(epochs, all_dfs)
    ]

    # join all the simulations
    df = all_dfs[0]
    for d in all_dfs[1:]:
        df = df.join(d)
    df = df.join(obs)

    outfile = run_dir / f"all_{run_dir.name}_results.csv"
    print(f"** ALL Results written to: {outfile}")
    df.to_csv(outfile)


if __name__ == "__main__":
    args = get_args()
    run_dir = Path(args["run_dir"])
    assert run_dir.exists()

    config_file = run_dir / "config.yml"
    cfg = read_config(config_file)

    analyse_all_epochs(cfg, run_dir)
