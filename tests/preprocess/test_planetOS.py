import xarray as xr
import numpy as np
from datetime import datetime
from pathlib import Path

from src.preprocess import PlanetOSPreprocessor
from src.utils import get_kenya, get_ethiopia

from ..utils import _make_dataset


class TestPlanetOSPreprocessor:
    @staticmethod
    def _make_era5POS_dataset(
        size, lonmin=0.0, lonmax=360.0, latmin=89.78, latmax=-89.78, add_times=True
    ):
        # Same as make_chirps_dataset, except with two variables
        # sine the era5 POS exporter might download multiple variables
        # also latitude -> lat, longitude -> lon, time -> time1
        lat_len, lon_len = size
        # create the vector
        longitudes = np.linspace(lonmin, lonmax, lon_len)
        latitudes = np.linspace(latmin, latmax, lat_len)

        dims = ["lon", "lat"]
        coords = {"lat": latitudes, "lon": longitudes}

        if add_times:
            size = (2, size[0], size[1])
            dims.insert(0, "time1")
            coords["time1"] = [datetime(2019, 1, 1), datetime(2019, 1, 2)]
        vhi = np.random.randint(100, size=size)
        precip = np.random.randint(100, size=size)

        return xr.Dataset({"VHI": (dims, vhi), "precip": (dims, precip)}, coords=coords)

    def test_init(self, tmp_path):

        PlanetOSPreprocessor(tmp_path)

        assert (tmp_path / "interim/era5POS_preprocessed").exists()
        assert (tmp_path / "interim/era5POS_interim").exists()

    @staticmethod
    def test_make_filename():
        path = Path("2008/01/vhi.nc")

        name = PlanetOSPreprocessor.create_filename(path, "kenya")
        expected_name = "2008_01_vhi_kenya.nc"
        assert name == expected_name, f"{name} generated, expected {expected_name}"

    @staticmethod
    def test_get_filenames(tmp_path):

        (tmp_path / "raw" / "era5POS").mkdir(parents=True)

        test_file = tmp_path / "raw/era5POS/testy_test.nc"
        test_file.touch()

        processor = PlanetOSPreprocessor(tmp_path)

        files = processor.get_filepaths()
        assert files[0] == test_file, f"Expected {test_file} to be retrieved"

    def test_preprocess(self, tmp_path):

        (tmp_path / "raw/era5POS/global").mkdir(parents=True)
        data_path = tmp_path / "raw/era5POS/global/testy_test.nc"
        dataset = self._make_era5POS_dataset(size=(100, 100))
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

        processor = PlanetOSPreprocessor(tmp_path)
        processor.preprocess(subset_str="kenya", regrid=regrid_path, n_processes=1)

        expected_out_path = tmp_path / "interim/era5POS_preprocessed/data_kenya.nc"
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

        assert out_data.VHI.values.shape[1:] == (20, 20)
        assert out_data.precip.values.shape[1:] == (20, 20)

        assert (
            not processor.interim.exists()
        ), f"Interim era5 folder should have been deleted"

    def test_rotate_and_filter(self):

        dataset = self._make_era5POS_dataset(size=(100, 100)).rename({"time1": "time"})
        rotated_ds = PlanetOSPreprocessor._rotate_and_filter(dataset)

        assert (rotated_ds.lon.min() > -180) & (
            rotated_ds.lon.max() < 180
        ), f"Longitudes not properly rotated!"

    def test_alternative_region(self, tmp_path):
        # make the dataset
        (tmp_path / "raw/era5POS/global").mkdir(parents=True)
        data_path = tmp_path / "raw/era5POS/global/testy_test.nc"
        dataset = self._make_era5POS_dataset(size=(100, 100))
        dataset.to_netcdf(path=data_path)
        ethiopia = get_ethiopia()

        regrid_dataset, _, _ = _make_dataset(
            size=(20, 20),
            latmin=ethiopia.latmin,
            latmax=ethiopia.latmax,
            lonmin=ethiopia.lonmin,
            lonmax=ethiopia.lonmax,
        )

        regrid_path = tmp_path / "regridder.nc"
        regrid_dataset.to_netcdf(regrid_path)

        processor = PlanetOSPreprocessor(tmp_path)
        processor.preprocess(subset_str="ethiopia", regrid=regrid_path, n_processes=1)

        expected_out_path = tmp_path / "interim/era5POS_preprocessed/data_ethiopia.nc"
        assert (
            expected_out_path.exists()
        ), f"Expected processed file to be saved to {expected_out_path}"
