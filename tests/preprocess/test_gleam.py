import xarray as xr
import numpy as np
from datetime import datetime

from src.preprocess import GLEAMPreprocessor
from src.utils import get_kenya, get_ethiopia

from ..utils import _make_dataset


class TestGLEAMPreprocessor:
    @staticmethod
    def test_make_filename():

        test_file = "testy_test.nc"
        expected_output = "testy_test_kenya.nc"

        filename = GLEAMPreprocessor.create_filename(test_file, "kenya")
        assert (
            filename == expected_output
        ), f"Expected output to be {expected_output}, got {filename}"

    @staticmethod
    def _make_gleam_dataset(size, lonmin=-180.0, lonmax=180.0, latmin=90, latmax=-90):
        lon_len, lat_len = size
        # create the vector
        longitudes = np.linspace(lonmin, lonmax, lon_len)
        latitudes = np.linspace(latmin, latmax, lat_len)

        dims_e = ["time", "lon", "lat"]
        dims_tb = ["time", "bnds"]

        coords = {"lat": latitudes, "lon": longitudes, "bnds": np.array([0, 1])}

        size = (2, size[0], size[1])
        coords["time"] = [datetime(2019, 1, 1), datetime(2019, 1, 2)]

        e = np.random.randint(100, size=size)
        time_bnds = np.random.randint(100, size=(2, len(coords["time"])))

        return xr.Dataset(
            {"E": (dims_e, e), "time_bnds": (dims_tb, time_bnds)}, coords=coords
        )

    @staticmethod
    def test_directories_created(tmp_path):
        v = GLEAMPreprocessor(tmp_path)

        assert (
            tmp_path / v.preprocessed_folder / "gleam_preprocessed"
        ).exists(), (
            "Should have created a directory tmp_path/interim/chirps_preprocessed"
        )

        assert (
            tmp_path / v.preprocessed_folder / "gleam_interim"
        ).exists(), "Should have created a directory tmp_path/interim/chirps_interim"

    @staticmethod
    def test_get_filenames(tmp_path):

        (tmp_path / "raw/gleam/monthly").mkdir(parents=True)

        test_file = tmp_path / "raw/gleam/monthly/testy_test.nc"
        test_file.touch()

        processor = GLEAMPreprocessor(tmp_path)

        files = processor.get_filepaths()
        assert files[0] == test_file, f"Expected {test_file} to be retrieved"

    def test_preprocess(self, tmp_path):

        (tmp_path / "raw/gleam/monthly").mkdir(parents=True)
        data_path = tmp_path / "raw/gleam/monthly/testy_test.nc"
        dataset = self._make_gleam_dataset(size=(100, 100))
        dataset.to_netcdf(path=data_path)

        kenya = get_kenya()
        regrid_dataset, _, _ = _make_dataset(
            size=(20, 20),
            latmin=kenya.latmin,
            latmax=kenya.latmax,
            lonmin=kenya.lonmin,
            lonmax=kenya.lonmax,
        )

        regrid_path = tmp_path / "regridder.nc"
        regrid_dataset.to_netcdf(regrid_path)

        processor = GLEAMPreprocessor(tmp_path)
        processor.preprocess(subset_str="kenya", regrid=regrid_path)

        expected_out_path = tmp_path / "interim/gleam_preprocessed/data_kenya.nc"
        assert (
            expected_out_path.exists()
        ), f"Expected processed file to be saved to {expected_out_path}"

        # check the subsetting happened correctly
        out_data = xr.open_dataset(expected_out_path)
        expected_dims = ["lat", "lon", "time"]
        assert len(list(out_data.dims)) == len(expected_dims)
        for dim in expected_dims:
            assert dim in list(
                out_data.dims
            ), f"Expected {dim} to be in the processed dataset dims"

        lons = out_data.lon.values
        assert (lons.min() >= kenya.lonmin) and (
            lons.max() <= kenya.lonmax
        ), "Longitudes not correctly subset"

        lats = out_data.lat.values
        assert (lats.min() >= kenya.latmin) and (
            lats.max() <= kenya.latmax
        ), "Latitudes not correctly subset"

        assert set(out_data.data_vars) == {"E"}, f"Got unexpected variables!"

        assert (
            not processor.interim.exists()
        ), f"Interim gleam folder should have been deleted"

    def test_swapaxes(self):

        dataset = self._make_gleam_dataset(size=(20, 30))

        out = GLEAMPreprocessor._swap_dims_and_filter(dataset)

        assert out.E.values.shape[1:] == (30, 20), f"Array axes not properly swapped!"

    def test_alternative_region(self, tmp_path):
        # make the dataset
        (tmp_path / "raw/gleam/monthly").mkdir(parents=True)
        data_path = tmp_path / "raw/gleam/monthly/testy_test.nc"
        dataset = self._make_gleam_dataset(size=(100, 100))
        dataset.to_netcdf(path=data_path)
        ethiopia = get_ethiopia()

        # regrid the datasets
        regrid_dataset, _, _ = _make_dataset(
            size=(20, 20),
            latmin=ethiopia.latmin,
            latmax=ethiopia.latmax,
            lonmin=ethiopia.lonmin,
            lonmax=ethiopia.lonmax,
        )
        regrid_path = tmp_path / "regridder.nc"
        regrid_dataset.to_netcdf(regrid_path)

        # build the Preprocessor object and subset with a different subset_str
        processor = GLEAMPreprocessor(tmp_path)
        processor.preprocess(subset_str="ethiopia", regrid=regrid_path)

        expected_out_path = tmp_path / "interim/gleam_preprocessed/data_ethiopia.nc"
        assert (
            expected_out_path.exists()
        ), f"Expected processed file to be saved to {expected_out_path}"
