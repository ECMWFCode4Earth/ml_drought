import xarray as xr
import numpy as np
import pandas as pd

from src.preprocess import VHIPreprocessor


class TestVHIPreprocessor:

    @staticmethod
    def test_vhi_init_directories_created(tmp_path):
        v = VHIPreprocessor(tmp_path)

        assert (tmp_path / v.preprocessed_folder / "vhi_preprocessed").exists(), \
            f'Should have created a directory tmp_path/interim/vhi_preprocessed'

        assert (tmp_path / v.preprocessed_folder / "vhi_interim").exists(), \
            f'Should have created a directory tmp_path/interim/vhi_interim'

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
        # add a file which shouldn't be added by get_vhi_filepaths
        (demo_raw_folder / 'info.md').touch()

        # get the filepaths using the VHIPreprocessor function
        recovered_names = [f.name for f in v.get_filepaths()]
        recovered_names.sort()

        assert recovered_names == fnames, \
            f'Expected all .nc files to be retrieved.'

    @staticmethod
    def test_preprocessor_output(tmp_path):
        v = VHIPreprocessor(tmp_path)

        # get filename
        demo_raw_folder = (v.raw_folder / 'vhi' / '1981')
        demo_raw_folder.mkdir(parents=True, exist_ok=True)
        netcdf_filepath = demo_raw_folder / 'VHP.G04.C07.NC.P1981035.VH.nc'

        # build dummy .nc object
        height = list(range(0, 3616))
        width = list(range(0, 10000))
        vci = tci = vhi = np.random.randint(100, size=(3616, 10000))

        raw_ds = xr.Dataset(
            {'VCI': (['HEIGHT', 'WIDTH'], vci),
             'TCI': (['HEIGHT', 'WIDTH'], tci),
             'VHI': (['HEIGHT', 'WIDTH'], vhi)},
            coords={
                'HEIGHT': height,
                'WIDTH': width}
        )
        raw_ds.to_netcdf(netcdf_filepath)

        # run the preprocessing steps
        out = v._preprocess(
            netcdf_filepath.as_posix(), v.interim.as_posix(),
        )
        expected = "STAR_VHP.G04.C07.NC_1981_8_31_kenya_VH.nc"

        assert out.name == expected, f"Expected: {expected} Got: {out.name}"

        # check the structure of the output file
        out_ds = xr.open_dataset(out)
        out_dims = list(out_ds.dims.keys())
        out_vars = [var for var in out_ds.variables.keys() if var not in out_dims]

        assert 'VHI' in out_vars, f'Expected to find VHI in out_vars'
        assert all(np.isin(['lat', 'lon', 'time'], out_dims)) \
            and (len(out_dims) == 3), \
            f'Expected {out_dims} to be ["lat","lon", "time"]'

    @staticmethod
    def test_make_filename():
        netcdf_filepath = 'VHP.G04.C07.NC.P1981035.VH.nc'
        subset_name = 'kenya'
        t = pd.to_datetime('1981-08-31')

        out_fname = VHIPreprocessor.create_vhi_filename(
            t,
            netcdf_filepath,
            subset_name,
        )

        expected = 'STAR_VHP.G04.C07.NC_1981_8_31_kenya_VH.nc'
        assert out_fname == expected, f'Expected: {expected}, got: {out_fname}'

    @staticmethod
    def test_create_new_dataset(tmp_path):

        netcdf_filepath = 'VHP.G04.C07.NC.P1981035.VH.nc'

        # build dummy .nc object
        height = list(range(0, 3616))
        width = list(range(0, 10000))
        vci = tci = vhi = np.random.randint(100, size=(3616, 10000))

        ds = xr.Dataset(
            {'VCI': (['HEIGHT', 'WIDTH'], vci),
             'TCI': (['HEIGHT', 'WIDTH'], tci),
             'VHI': (['HEIGHT', 'WIDTH'], vhi)},
            coords={
                'HEIGHT': height,
                'WIDTH': width}
        )

        processor = VHIPreprocessor(tmp_path)
        timestamp = processor.extract_timestamp(ds, netcdf_filepath, use_filepath=True)
        expected_timestamp = pd.Timestamp('1981-08-31 00:00:00')

        assert timestamp == expected_timestamp, f"Timestamp. \
            Expected: {expected_timestamp} Got: {timestamp}"

        longitudes, latitudes = processor.create_lat_lon_vectors(ds)
        exp_long = np.linspace(-180, 180, 10000)
        assert all(longitudes == exp_long), f"Longitudes \
            not what expected: np.linspace(-180,180,10000)"

        exp_lat = np.linspace(-55.152, 75.024, 3616)
        assert all(latitudes == exp_lat), f"latitudes \
            not what expected: np.linspace(-55.152,75.024,3616)"

        out_ds = processor.create_new_dataset(ds,
                                              longitudes,
                                              latitudes,
                                              timestamp,
                                              all_vars=False)

        assert isinstance(out_ds, xr.Dataset), \
            f'Expected out_ds to be of type: xr.Dataset, now: {type(out_ds)}'

        # test a time dimension was added
        out_dims = list(out_ds.dims.keys())
        assert 'time' in out_dims, f'Expected "time" to be in dataset coords, got {out_dims}'

        # test the other variables were removed
        out_variables = [var for var in out_ds.variables.keys() if var not in out_dims]
        assert out_variables == ['VHI'], \
            f'Expected dataset variables to only have VHI, got {out_variables}'
