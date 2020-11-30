import xarray as xr
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import numpy as np

import sys
import subprocess

sys.path.insert(2, "/home/tommy/neuralhydrology")
# sys.path.insert(2, "/Users/tommylees/github/neuralhydrology")
from neuralhydrology.utils.config import Config


def _get_trained_model_dir(config) -> Path:
    #  TODO: currently selecting the most recent (latest in terms of training time)
    trained_model_dirs = list(config.run_dir.glob(f"{config.experiment_name}*"))
    assert len(trained_model_dirs) > 0, "Expect the model to already be trained!"
    trained_model_dir = trained_model_dirs[-1]
    return trained_model_dir


def _create_run_dir_for_expt(finetune_basin: int, data_dir: Path) -> Path:
    run_dir = Path(data_dir / f"runs/ensemble_finetune/ALL_FINE/FINE_{finetune_basin}")
    run_dir.mkdir(exist_ok=True, parents=True)
    return run_dir


def _create_config_directory(parent_config_path: Path, finetune_basin: int) -> Path:
    # make the config_dir
    config_dir = parent_config_path.parents[1] / f"FINE_{finetune_basin}"
    config_dir.mkdir(exist_ok=True, parents=True)
    return config_dir


def create_basin_txt_file(finetune_basin: int, base_dir: Path) -> Path:
    basin_txt_file = f"FINE_{finetune_basin}.txt"
    basin_txt_path = base_dir / "neuralhydrology/data" / basin_txt_file
    assert (
        basin_txt_path.exists()
    ), f"Expect the data path: {basin_txt_path.as_posix()} to exist"

    with open(basin_txt_path, "w+") as f:
        f.write(str(finetune_basin))

    return basin_txt_path


def create_fine_tuning_config(
    parent_config_path: Path,
    finetune_basin: int,
    output_config_dir: Path,
    run_dir: Path,
    basin_txt_path: Path,
) -> Path:
    #  load the parent config file
    config = Config(parent_config_path)

    # get the trained_model_dir from the config (for this ensemble member)
    trained_model_dir = _get_trained_model_dir(config)

    # -----------------
    # MAKE CONFIG FILE
    # -----------------
    experiment_name = f"FINE_{finetune_basin}_{config.experiment_name}"

    # update config args
    config.force_update(
        dict(
            is_finetuning=True,
            finetune_modules=["head"],
            train_start_date="01/10/1970",
            run_dir=run_dir,
            epochs=20,
            base_run_dir=trained_model_dir,
            validate_every=3,
            train_basin_file=basin_txt_path,
            validation_basin_file=basin_txt_path,
            test_basin_file=basin_txt_path,
            experiment_name=experiment_name,
        )
    )

    # save the config with updated features
    filename = f"FINE_{finetune_basin}_{parent_config_path.name}"
    output_config_path = output_config_dir / filename
    if output_config_path.exists():
        print(f"Config file already exists at: {output_config_path}. Overwriting")
        output_config_path.unlink()

    config.dump_config(folder=output_config_dir, filename=filename)

    return output_config_path


def setup_configs_for_experiment(
    data_dir: Path, base_config_dir: Path, finetune_basin: int,
) -> Path:
    #  load in config files
    config_paths = list(base_config_dir.glob("*"))

    #  create the folders for this basin experiment
    output_config_dir = _create_config_directory(config_paths[0], finetune_basin)
    run_dir = _create_run_dir_for_expt(finetune_basin, data_dir)
    basin_txt_path = create_basin_txt_file(finetune_basin, base_dir)

    # parent_config_path = config_paths[0]
    for parent_config_path in config_paths:
        create_fine_tuning_config(
            parent_config_path=parent_config_path,
            finetune_basin=finetune_basin,
            run_dir=run_dir,
            output_config_dir=output_config_dir,
            basin_txt_path=basin_txt_path,
        )

    return run_dir, output_config_dir


def run_finetune_training(base_dir: Path, output_config_dir: Path):
    """Run the finetuning training in parallel using the run_scheduler script
        ipython --pdb neuralhydrology/nh_run_scheduler.py finetune -- /
            --directory configs/ensemble_lstm_finetune /
            --runs-per-gpu 2 --gpu-ids 0
    """
    print(f"\n\n ** Finetune Training: {output_config_dir.as_posix()} ** \n\n")
    p = subprocess.run(
        [
            "ipython",
            "neuralhydrology/nh_run_scheduler.py",
            "finetune",
            "--",
            "--directory",
            output_config_dir.as_posix(),
            "--runs-per-gpu",
            "2",
            "--gpu-ids",
            "0",
        ],
        cwd=base_dir / "neuralhydrology",
    )
    assert p.returncode == 0


def run_finetune_evaluate(base_dir: Path, run_dir: Path):
    """Finetune evaluate results for ensemble members
        ipython --pdb neuralhydrology/nh_run_scheduler.py evaluate -- /
            --directory /cats/datastore/data/runs/ensemble_finetune/FINE /
            --runs-per-gpu 2 --gpu-ids 0
    """
    print(
        f"\n\n ** Finetune Evaluation (ensemble members): {run_dir.as_posix()} ** \n\n"
    )
    p = subprocess.run(
        [
            "ipython",
            "neuralhydrology/nh_run_scheduler.py",
            "evaluate",
            "--",
            "--directory",
            run_dir.as_posix(),
            "--runs-per-gpu",
            "2",
            "--gpu-ids",
            "0",
        ],
        cwd=base_dir / "neuralhydrology",
    )
    assert p.returncode == 0


def run_finetune_merge(base_dir: Path, run_dir: Path):
    """FINETUNE merge results of ensemble members
        ipython --pdb neuralhydrology/utils/nh_results_ensemble.py -- /
        --run-dirs /cats/datastore/data/runs/ensemble_finetune/FINE/* /
        --save-file cats/datastore/data/runs/ensemble_finetune/FINE/ensemble_results.p /
        --metrics NSE MSE KGE FHV FMS FLV`
    """
    print(f"\n\n ** Finetune Merge Ensemble Members: {run_dir.as_posix()} ** \n\n")

    all_run_dirs = [p.as_posix() for p in list(run_dir.glob("*"))]
    metrics = "NSE MSE KGE FHV FMS FLV".split(" ")
    p = subprocess.run(
        [
            "ipython",
            "--pdb",
            "neuralhydrology/utils/nh_results_ensemble.py",
            "--",
            "--run-dirs",
            *all_run_dirs,
            "--save-file",
            f"{run_dir.as_posix()}/ensemble_results.p",
            "--metrics",
            *metrics,
        ],
        cwd=base_dir / "neuralhydrology",
    )
    assert p.returncode == 0


def run_finetune_get_results(base_dir: Path, run_dir: Path):
    """Finetune Write results Out to .csv/.nc
        `cd /home/tommy/tommy_multiple_forcing; ipython --pdb analysis/read_nh_results.py -- --run_dir /cats/datastore/data/runs/ensemble_finetune/FINE/ --ensemble True --ensemble_filename /cats/datastore/data/runs/ensemble_finetune/FINE/ensemble_results.p; cd -`
    """
    output_file = run_dir / "ensemble_results.p"
    p = subprocess.run(
        [
            "ipython",
            "--pdb",
            # "analysis/read_nh_results.py",
            "scripts/drafts/multiple_forcing/read_nh_results.py",
            "--",
            "--run_dir",
            run_dir.as_posix(),
            "--ensemble",
            "True",
            "--ensemble_filename",
            output_file.as_posix(),
        ],
        cwd=base_dir / "ml_drought",
    )
    assert p.returncode == 0


if __name__ == "__main__":
    #  CREATE AND TRAIN AN ENSEMBLE OF FINE-TUNED MODELS FOR ONE BASIN
    base_dir = Path("/home/tommy/")
    data_dir = Path("/cats/datastore/data")
    base_config_dir = base_dir / "neuralhydrology/configs/ensemble_lstm"
    finetune_basin = 54052  #  41004 41019

    assert data_dir.exists()
    assert base_dir.exists()
    assert base_config_dir.exists()

    run_dir, output_config_dir = setup_configs_for_experiment(
        data_dir=data_dir,
        base_config_dir=base_config_dir,
        finetune_basin=finetune_basin,
    )
    print(f"Finished Setting up Experiment. Configs are stored in {output_config_dir}")

    # run the analysis functions / scripts
    # train -> evaluate -> merge -> get results
    run_finetune_training(base_dir, output_config_dir)
    run_finetune_evaluate(base_dir, run_dir)
    run_finetune_merge(base_dir, run_dir)
    run_finetune_get_results(base_dir, run_dir)

    # get the results
    list(run_dir.glob("*ENS.csv"))[0]
    list(run_dir.glob("metric_df.csv"))[0]
    list(run_dir.glob("results.nc"))[0]

