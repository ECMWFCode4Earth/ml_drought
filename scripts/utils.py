import shutil
from pathlib import Path
import time
import json


def get_data_path() -> Path:
    # if the working directory is alread ml_drought don't need ../data
    if Path(".").absolute().as_posix().split("/")[-1] == "ml_drought":
        data_path = Path("data")
    elif Path(".").absolute().as_posix().split("/")[-3] == "ml_drought":
        data_path = Path("../../data")
    else:
        data_path = Path("../data")
    return data_path


def _rename_directory(
    from_path: Path, to_path: Path, with_datetime: bool = False
) -> None:
    if with_datetime:
        dt = time.gmtime()
        dt_str = f"{dt.tm_year}_{dt.tm_mon:02}_{dt.tm_mday:02}:{dt.tm_hour:02}{dt.tm_min:02}{dt.tm_sec:02}"
        name = "/" + dt_str + "_" + to_path.as_posix().split("/")[-1]
        to_path = "/".join(to_path.as_posix().split("/")[:-1]) + name
        to_path = Path(to_path)
    shutil.move(from_path.as_posix(), to_path.as_posix())
    print(f"MOVED {from_path} to {to_path}")


def get_results(dir_: Path):
    """ Get the results from the results.json """

    regex = r'\d{4}_\d{2}_\d{2}:\d{6}_'

    result_paths = [p for p in dir_.glob('*/*/results.json')]
    experiments = [

    ]
    models = [p.parents[0].name for p in result_paths]
    result_dicts = [json.load(p) for p in result_paths]
    total_rmse = [d['total'] for d in result_dicts]

    for i, experiment in enumerate(experiments):
        print(
            f"Experiment: {experiment}\n"
            f"Model: {models[i]}\n"
            f"Persistence RMSE: {}\n"
            f"RMSE: {}\n"
        )

