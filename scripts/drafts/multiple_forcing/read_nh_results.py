"""To be run after training and evaluation

1) Train
    `ipython --pdb neuralhydrology/nh_run.py train -- --config-file /path/to/config`
2) Evaluation
    `ipython --pdb neuralhydrology/nh_run.py evaluate -- --run-dir /path/to/run_dir --metrics NSE MSE KGE FHV FMS FLV`
3) Extract Results
    `ipython --pdb analysis/read_nh_results.py -- --run_dir /path/to/run_dir --epoch 30`

## ENSEMBLE
1) Train
    `ipython --pdb neuralhydrology/nh_run_scheduler.py train -- --directory configs/ensemble_ealstm_LANE/ --gpu-ids 0 --runs-per-gpu 2`
2) Evaluate
    `ipython --pdb neuralhydrology/nh_run_scheduler.py evaluate -- --directory /cats/datastore/data/runs/ensemble_ealstm_LANE --runs-per-gpu 2 --gpu-ids 0`
3) Merge Results
    `ipython --pdb neuralhydrology/utils/nh_results_ensemble.py -- --run-dirs /cats/datastore/data/runs/ensemble_ealstm_LANE/*  --save-file /cats/datastore/data/runs/ensemble_ealstm_LANE/ensemble_results.p --metrics NSE MSE KGE FHV FMS FLV`
4) Extract Results
    `cd /home/tommy/ml_drought; ipython --pdb scripts/drafts/multiple_forcing/read_nh_results.py -- --run_dir /cats/datastore/data/runs/ensemble_ealstm_LANE --ensemble True --ensemble_filename /cats/datastore/data/runs/ensemble_ealstm_LANE/ensemble_results.p`
•) Read each individual member
    

5) Finetune
    `ipython --pdb neuralhydrology/nh_run_scheduler.py finetune -- --directory configs/ensemble_lstm_finetune/ --runs-per-gpu 2 --gpu-ids 0`
6) Finetune evaluate
    `ipython --pdb neuralhydrology/nh_run_scheduler.py evaluate -- --directory /cats/datastore/data/runs/ensemble_finetune/FINE --runs-per-gpu 2 --gpu-ids 0`
7) FINETUNE merge results
    `ipython --pdb neuralhydrology/utils/nh_results_ensemble.py -- --run-dirs /cats/datastore/data/runs/ensemble_finetune/FINE/* --save-file /cats/datastore/data/runs/ensemble_finetune/FINE/ensemble_results.p --metrics NSE MSE KGE FHV FMS FLV`
8) Finetune results
    `cd /home/tommy/tommy_multiple_forcing; ipython --pdb analysis/read_nh_results.py -- --run_dir /cats/datastore/data/runs/ensemble_finetune/FINE/ --ensemble True --ensemble_filename /cats/datastore/data/runs/ensemble_finetune/FINE/ensemble_results.p; cd -`
"""
import xarray as xr
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
from typing import Tuple, Dict, Optional, DefaultDict, Union
from collections import defaultdict
from tqdm import tqdm

# running evaluation
import sys

sys.path.append("/home/tommy/neuralhydrology")
from neuralhydrology.evaluation import RegressionTester as Tester
from neuralhydrology.utils.config import Config


def run_evaluation(run_dir: Path, epoch: Optional[int] = None, period: str = "test"):
    """Helper Function to run the evaluation
    (same as def start_evaluation: neuralhydrology/evaluation/evaluate.py:L7)

    Args:
        run_dir (Path): Path of the experiment run
        epoch (Optional[int], optional):
            Model epoch to evaluate. None finds the latest (highest) epoch.
            Defaults to None.
        period (str, optional): {"test", "train", "validation"}. Defaults to "test".
    """
    #
    cfg = Config(run_dir / "config.yml")
    tester = Tester(cfg=cfg, run_dir=run_dir, period=period, init_model=True)

    if epoch is None:
        # get the highest epoch trained model
        all_trained_models = [d.name for d in (run_dir).glob("model_epoch*.pt")]
        epoch = int(
            sorted(all_trained_models)[-1].replace(".pt", "").replace("model_epoch", "")
        )
    print(f"** EVALUATING MODEL EPOCH: {epoch} **")
    tester.evaluate(epoch=epoch, save_results=True, metrics=["NSE", "KGE"])


def get_args() -> Dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str)
    parser.add_argument("--eval", type=bool, default=False)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--save_csv", type=bool, default=True)
    parser.add_argument("--ensemble", type=bool, default=False)
    parser.add_argument("--ensemble_filename", type=str, default=None)
    args = vars(parser.parse_args())

    return args


def get_test_filepath(run_dir: Path, epoch: Optional[int]) -> Path:
    # create filepath for test
    if epoch is None:
        #  get the maximum epoch
        all_evaluated_results = [d.name for d in (run_dir / "test").iterdir()]
        epoch = int(sorted(all_evaluated_results)[-1].replace("model_epoch", ""))

    test_dir = run_dir / f"test/model_epoch{epoch:03}/"
    res_fp = test_dir / "test_results.p"

    assert (
        res_fp.exists()
    ), "Has validation been run? ipython --pdb neuralhydrology/nh_run.py evaluate -- --run-dir /path/to/run_dir"

    return res_fp


def get_ensemble_path(
    run_dir: Path, ensemble_filename: str = "ensemble_results.p"
) -> Path:
    res_fp = run_dir / ensemble_filename
    assert res_fp.exists(), f"Has validation been run? I cannot find {res_fp}"
    return res_fp


def get_ds_and_metrics(res_fp: Path) -> Union[xr.Dataset, pd.DataFrame]:
    # load the dictionary of results
    res_dict = pickle.load(res_fp.open("rb"))
    stations = [k for k in res_dict.keys()]

    # should only contain one frequency
    freq = [k for k in res_dict[stations[0]].keys()]
    assert len(freq) == 1
    freq = freq[0]

    #  Create List of Datasets (obs, sim) and metric DataFrame
    output_metrics_dict: DefaultDict[str, List] = defaultdict(list)
    all_xr_objects: List[xr.Dataset] = []

    for station_id in tqdm(stations):
        #  extract the raw results
        try:
            xr_obj = (
                res_dict[station_id][freq]["xr"].isel(time_step=0).drop("time_step")
            )
        except ValueError:
            # ensemble mode does not have "time_step" dimension
            xr_obj = res_dict[station_id][freq]["xr"].rename({"datetime": "date"})
        xr_obj = xr_obj.expand_dims({"station_id": [station_id]}).rename(
            {"date": "time"}
        )
        all_xr_objects.append(xr_obj)

        # extract the output metrics
        output_metrics_dict["station_id"].append(station_id)
        try:
            output_metrics_dict["NSE"].append(res_dict[station_id][freq]["NSE"])
            output_metrics_dict["KGE"].append(res_dict[station_id][freq]["KGE"])
            output_metrics_dict["MSE"].append(res_dict[station_id][freq]["MSE"])
            output_metrics_dict["FHV"].append(res_dict[station_id][freq]["FHV"])
            output_metrics_dict["FMS"].append(res_dict[station_id][freq]["FMS"])
            output_metrics_dict["FLV"].append(res_dict[station_id][freq]["FLV"])
        except KeyError:
            try:
                output_metrics_dict["NSE"].append(
                    res_dict[station_id][freq][f"NSE_{freq}"]
                )
                output_metrics_dict["KGE"].append(
                    res_dict[station_id][freq][f"KGE_{freq}"]
                )
                output_metrics_dict["MSE"].append(
                    res_dict[station_id][freq][f"MSE_{freq}"]
                )
                output_metrics_dict["FHV"].append(
                    res_dict[station_id][freq][f"FHV_{freq}"]
                )
                output_metrics_dict["FMS"].append(
                    res_dict[station_id][freq][f"FMS_{freq}"]
                )
                output_metrics_dict["FLV"].append(
                    res_dict[station_id][freq][f"FLV_{freq}"]
                )
            except KeyError:
                output_metrics_dict["NSE"].append(np.nan)
                output_metrics_dict["KGE"].append(np.nan)
                output_metrics_dict["MSE"].append(np.nan)
                output_metrics_dict["FHV"].append(np.nan)
                output_metrics_dict["FMS"].append(np.nan)
                output_metrics_dict["FLV"].append(np.nan)

    #  merge all stations into one xarray object
    ds = xr.concat(all_xr_objects, dim="station_id")

    #  create metric dataframe
    metric_df = pd.DataFrame(output_metrics_dict)

    return ds, metric_df


def get_old_format_dataframe(ds: xr.Dataset) -> pd.DataFrame:
    df = (
        ds.to_dataframe()
        .reset_index()
        .rename({"discharge_spec_obs": "obs", "discharge_spec_sim": "sim"}, axis=1)
    )
    return df


if __name__ == "__main__":
    #  read cmd line arguments
    args = get_args()
    bool_evaluation: bool = args["eval"]
    run_dir: Path = Path(args["run_dir"])
    epoch: Optional[int] = args["epoch"]
    save_csv: bool = args["save_csv"]
    ensemble: bool = args["ensemble"]
    ensemble_filename: str = args["ensemble_filename"]

    # run evaluation (optional)
    if bool_evaluation:
        run_evaluation(run_dir, epoch=epoch, period="test")

    if ensemble:
        res_fp = get_ensemble_path(run_dir, ensemble_filename)
    else:
        res_fp = get_test_filepath(run_dir, epoch)
    test_dir = res_fp.parents[0]

    ds, metric_df = get_ds_and_metrics(res_fp)

    # SAVE
    print("** Writing `results.nc` and `metric_df.csv` **")
    ds.to_netcdf(test_dir / "results.nc")
    metric_df.to_csv(test_dir / "metric_df.csv")

    # create csv with informative name
    if save_csv:
        print("** Writing results as .csv file **")
        try:
            fname = f"{test_dir.parents[1].name}_E{epoch:03}.csv"
        except:
            fname = f"{test_dir.parents[1].name}_ENS.csv"
        df = get_old_format_dataframe(ds)
        df.to_csv(test_dir / fname)
