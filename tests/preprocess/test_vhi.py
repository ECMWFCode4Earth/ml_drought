import xarray as xr
import numpy as np
import pandas as pd

from src.preprocess.vhi import (
    VHIPreprocessor,
    create_filename,
    extract_timestamp,
    create_lat_lon_vectors,
    create_new_dataset
)


class TestVHIPreprocessor:

    @staticmethod
    def test_vhi_init_directories_created(tmp_path):
        v = VHIPreprocessor(tmp_path)

        assert (tmp_path / v.interim_folder / "vhi_preprocessed").exists(), f"\
            Should have created a directory tmp_path/interim/vhi_preprocessed"

        assert (tmp_path / v.interim_folder / "vhi").exists(), f"\
            Should have created a directory tmp_path/interim/vhi"

    @staticmethod
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

        assert recovered_names == fnames, \
            f'Recovered filenames should be the same as those created'

    @staticmethod
    def test_preprocessor_output(tmp_path):
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

        # run the preprocessing steps
        out = v.preprocess_vhi_data(
            netcdf_filepath.as_posix(), v.vhi_interim.as_posix(),
        )
        expected = "STAR_VHP.G04.C07.NC_1981_8_31_kenya_VH.nc"

        assert "kenya_VH.nc" in out.name, f"Expected to find kenya_VH.nc \
            in output path name but found: {out.name}"
        assert out.name == expected, f"Expected: {expected} Got: {out.name}"

        # check the structure of the output file
        out_ds = xr.open_dataset(out)
        out_dims = [dim for dim in out_ds.dims.keys()]
        out_vars = [var for var in out_ds.variables.keys() if var not in out_dims]

        assert 'VHI' in out_vars, f"Expected to find VHI in out_vars"
        assert all(np.isin(['latitude', 'longitude', 'time'], out_dims)), f"Expected\
            to find ['latitude','longitude', 'time'] in {out_dims}"

    @staticmethod
    def test_make_filename():
        netcdf_filepath = 'VHP.G04.C07.NC.P1981035.VH.nc'
        subset = True
        subset_name = 'kenya'
        t = pd.to_datetime('1981-08-31')

        out_fname = create_filename(
            t,
            netcdf_filepath,
            subset,
            subset_name,
        )

        expected = 'STAR_VHP.G04.C07.NC_1981_8_31_kenya_VH.nc'
        assert out_fname == expected, f"\nExpected: {expected} \nGot: {out_fname}"

    @staticmethod
    def test_create_new_dataset():

        netcdf_filepath = 'VHP.G04.C07.NC.P1981035.VH.nc'

        # build dummy .nc object
        HEIGHT = list(range(0, 3616))
        WIDTH = list(range(0, 10000))
        VCI = TCI = VHI = np.random.randint(100, size=(3616, 10000))

        ds = xr.Dataset(
            {'VCI': (['HEIGHT', 'WIDTH'], VCI),
             'TCI': (['HEIGHT', 'WIDTH'], TCI),
             'VHI': (['HEIGHT', 'WIDTH'], VHI)},
            coords={
                'HEIGHT': (HEIGHT),
                'WIDTH': (WIDTH), }
        )
        timestamp = extract_timestamp(ds, netcdf_filepath, use_filepath=True)
        expected_timestamp = pd.Timestamp('1981-08-31 00:00:00')

        assert timestamp == expected_timestamp, f"Timestamp. \
            Expected: {expected_timestamp} Got: {timestamp}"

        longitudes, latitudes = create_lat_lon_vectors(ds)
        exp_long = np.linspace(-180, 180, 10000)
        assert all(longitudes == exp_long), f"Longitudes \
            not what expected: np.linspace(-180,180,10000)"

        exp_lat = np.linspace(-55.152, 75.024, 3616)
        assert all(latitudes == exp_lat), f"latitudes \
            not what expected: np.linspace(-55.152,75.024,3616)"

        out_ds = create_new_dataset(ds,
                                    longitudes,
                                    latitudes,
                                    timestamp,
                                    all_vars=False)

        assert isinstance(out_ds, xr.Dataset), f"Expected out_ds to be of \
            type: xr.Dataset, now: {type(out_ds)}"
