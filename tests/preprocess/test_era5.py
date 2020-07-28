import xarray as xr
import numpy as np

# from datetime import datetime
import pandas as pd
from pathlib import Path
import pytest

from src.preprocess import ERA5MonthlyMeanPreprocessor, ERA5HourlyPreprocessor
from src.utils import get_kenya

from ..utils import _make_dataset


class TestPlanetOSPreprocessor:
    @staticmethod
    def _make_era5_dataset(
        size,
        lonmin=33.75,
        lonmax=42.25,
        latmin=6.0,
        latmax=-5.0,
        add_times=True,
        monthly=True,
        min_date="2019-01-01",
        max_date="2019-03-01",
    ):
        # Same as make_chirps_dataset, except already truncated
        # since we can just download Kenya from the cds api
        lat_len, lon_len = size
        # create the vector
        longitudes = np.linspace(lonmin, lonmax, lon_len)
        latitudes = np.linspace(latmin, latmax, lat_len)

        dims = ["longitude", "latitude"]
        coords = {"latitude": latitudes, "longitude": longitudes}

        if add_times:
            dims.insert(0, "time")

            if monthly:
                times = pd.date_range(min_date, max_date, freq="M")
            else:
                times = pd.date_range(min_date, max_date, freq="H")[:10]

            size = (len(times), size[0], size[1])
            coords["time"] = times
        t2m = np.random.randint(100, size=size)

        return xr.Dataset({"t2m": (dims, t2m)}, coords=coords)

    def test_init(self, tmp_path):

        ERA5MonthlyMeanPreprocessor(tmp_path)

        assert (
            tmp_path / "interim/reanalysis-era5-single-levels-"
            "monthly-means_preprocessed"
        ).exists()
        assert (
            tmp_path / "interim/reanalysis-era5-single-levels-" "monthly-means_interim"
        ).exists()

    def test_init_hourly(self, tmp_path):

        ERA5HourlyPreprocessor(tmp_path)

        assert (
            tmp_path / "interim/reanalysis-era5-single-levels_preprocessed"
        ).exists()
        assert (tmp_path / "interim/reanalysis-era5-single-levels_interim").exists()

    @staticmethod
    @pytest.mark.parametrize("granularity", [("hourly"), ("monthly")])
    def test_make_filename(tmp_path, granularity):
        if granularity == "monthly":
            basename = "reanalysis-era5-single-levels-monthly-means"
            processor = ERA5MonthlyMeanPreprocessor(tmp_path)
        elif granularity == "hourly":
            basename = "reanalysis-era5-single-levels"
            processor = ERA5HourlyPreprocessor(tmp_path)

        path = Path(basename + "/2m_temperature/1979_2019/01_12.nc")

        name = processor.create_filename(path, "kenya")
        expected_name = "1979_2019_01_12_2m_temperature_kenya.nc"
        assert name == expected_name, f"{name} generated, expected {expected_name}"

    @staticmethod
    @pytest.mark.parametrize("granularity", [("hourly"), ("monthly")])
    def test_get_filenames(tmp_path, granularity):

        if granularity == "monthly":
            basename = "reanalysis-era5-single-levels-monthly-means"
            processor = ERA5MonthlyMeanPreprocessor(tmp_path)
        elif granularity == "hourly":
            basename = "reanalysis-era5-single-levels"
            processor = ERA5HourlyPreprocessor(tmp_path)
        (tmp_path / f"raw/{basename}/" "2m_temperature/1979_2019").mkdir(parents=True)

        test_file = tmp_path / f"raw/{basename}/" "/2m_temperature/1979_2019.01_12.nc"
        test_file.touch()

        files = processor.get_filepaths()
        assert files[0] == test_file, f"Expected {test_file} to be retrieved"

    @pytest.mark.parametrize("granularity", [("hourly"), ("monthly")])
    def test_preprocess(self, tmp_path, granularity):
        if granularity == "monthly":
            basename = "reanalysis-era5-single-levels-monthly-means"
            processor = ERA5MonthlyMeanPreprocessor(tmp_path)
        elif granularity == "hourly":
            basename = "reanalysis-era5-single-levels"
            processor = ERA5HourlyPreprocessor(tmp_path)

        (tmp_path / f"raw/{basename}/" "2m_temperature/1979_2019").mkdir(parents=True)
        data_path = tmp_path / f"raw/{basename}/" "2m_temperature/1979_2019/01_12.nc"
        if granularity == "hourly":
            dataset = self._make_era5_dataset(size=(100, 100), monthly=False)
        else:
            dataset = self._make_era5_dataset(size=(100, 100), monthly=True)
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

        processor.preprocess(subset_str="kenya", regrid=regrid_path, n_processes=1)
        if granularity == "monthly":
            expected_out_path = (
                tmp_path / f"interim/{basename}" "_preprocessed/data_kenya.nc"
            )
        else:
            expected_out_path = (
                tmp_path / f"interim/{basename}"
                "_preprocessed/hourly_2m_temperature_data_kenya.nc"
            )

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

        assert out_data.t2m.values.shape[1:] == (20, 20)

        assert (
            not processor.interim.exists()
        ), f"Interim era5 folder should have been deleted"
