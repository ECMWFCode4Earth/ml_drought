from pathlib import Path

from src.preprocess.vhi import (
    VHIPreprocessor,
)

# get the root project directory
project_dir = Path(__file__).resolve().parents[2]


def test_vhi_init_directories_created(tmp_path):
    v = VHIPreprocessor(tmp_path)

    assert (tmp_path / v.interim_folder / "vhi_preprocessed").exists(), f"\
        Should have created a directory tmp_path/interim/vhi_preprocessed"

    assert (tmp_path / v.interim_folder / "vhi").exists(), f"\
        Should have created a directory tmp_path/interim/vhi"


def test_vhi_raw_filenames():
    pass
