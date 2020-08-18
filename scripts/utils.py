import shutil
from pathlib import Path
import time


def get_data_path() -> Path:
    # if the working directory is alread ml_drought don't need ../data
    if "/home/tommy" in Path(".").absolute().as_posix():
        # on AWS machine
        data_path = Path("/cats/datastore/data")
        if not data_path.exists():
            data_path.mkdir(parents=True, exist_ok=True)
    else:
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


def _base_increment_folder(data_path: Path, dir: str):
    old_paths = [d for d in data_path.glob(f"*_{dir}*")]
    if old_paths == []:
        integer = 0
    else:
        old_max = max([int(p.name.split("_")[0]) for p in old_paths])
        integer = old_max + 1

    _rename_directory(
        from_path=data_path / f"{dir}",
        to_path=data_path / f"{integer}_{dir}",
        with_datetime=False,
    )


def rename_features_dir(data_path: Path, dir: str):
    """increment the features dir by 1"""
    _base_increment_folder(data_path, dir="features")


def rename_models_dir(data_path: Path):
    """increment the models dir by 1"""
    _base_increment_folder(data_path, dir="models")
