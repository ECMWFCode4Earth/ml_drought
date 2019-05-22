from pathlib import Path
import xarray as xr
import numpy as np

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


def test_vhi_raw_filenames(tmp_path):
    v = VHIPreprocessor(tmp_path)
    demo_raw_folder = (v.raw_folder / 'vhi' / '1981')
    demo_raw_folder.mkdir(parents=True, exist_ok=True)

    fnames = ['VHP.G04.C07.NC.P1981035.VH.nc',
              'VHP.G04.C07.NC.P1981036.VH.nc',
              'VHP.G04.C07.NC.P1981037.VH.nc',
              'VHP.G04.C07.NC.P1981038.VH.nc',
              'VHP.G04.C07.NC.P1981039.VH.nc']

    # touch placeholder files
    [(demo_raw_folder / fname).touch() for fname in fnames]

    # get the filepaths using the VHIPreprocessor function
    recovered_names = [f.name for f in v.get_vhi_filepaths()]
    recovered_names.sort()

    assert recovered_names == fnames, f"Recovered filenames should be the same as those created "


def test_(tmp_path):
    v = VHIPreprocessor(tmp_path)

    # get filename
    demo_raw_folder = (v.raw_folder / 'vhi' / '1981')
    demo_raw_folder.mkdir(parents=True, exist_ok=True)
    netcdf_filepath = demo_raw_folder / 'VHP.G04.C07.NC.P1981035.VH.nc'

    # build dummy .nc object
    HEIGHT = list(range(0, 3616))
    WIDTH = list(range(0, 10000))
    VCI = TCI = VHI = np.random.randint(100, size=(3616, 10000))

    raw_ds = xr.Dataset(
        {'VCI': (['HEIGHT', 'WIDTH'], VCI),
         'TCI': (['HEIGHT', 'WIDTH'], TCI),
         'VHI': (['HEIGHT', 'WIDTH'], VHI)},
        coords={
            'HEIGHT': (HEIGHT),
            'WIDTH': (WIDTH), }
    )
    raw_ds.to_netcdf(netcdf_filepath)

    # v.preprocess_VHI_data(netcdf_filepath.as_posix(), v.vhi_interim.as_posix())
    v.add_coordinates(netcdf_filepath, subset='kenya')
