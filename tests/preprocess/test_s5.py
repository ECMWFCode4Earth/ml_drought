from pathlib import Path
import xarray as xr
import numpy as np
import pandas as pd
import cfgrib

from src.preprocess.seas5.ouce_s5 import OuceS5Data
from src.preprocess import S5Preprocessor
from src.utils import get_kenya
from ..utils import _make_dataset


def make_dummy_seas5_data(date_str: str) -> xr.Dataset:
    initialisation_date = pd.date_range(start=date_str, periods=1, freq="M")
    number = [i for i in range(0, 26)]  # corresponds to ensemble number (51)
    lat = np.linspace(-5.175003, -5.202, 36)
    lon = np.linspace(33.5, 42.25, 45)
    forecast_horizon = np.array(
        [
            2419200000000000,
            2592000000000000,
            2678400000000000,
            5097600000000000,
            5270400000000000,
            5356800000000000,
            7689600000000000,
            7776000000000000,
            7862400000000000,
            7948800000000000,
            10368000000000000,
            10454400000000000,
            10540800000000000,
            10627200000000000,
            12960000000000000,
            13046400000000000,
            13219200000000000,
            15638400000000000,
            15724800000000000,
            15811200000000000,
            15897600000000000,
            18316800000000000,
            18489600000000000,
            18576000000000000,
        ],
        dtype="timedelta64[ns]",
    )
    valid_time = initialisation_date[:, np.newaxis] + forecast_horizon
    precip = np.ones(
        shape=(
            len(number),
            len(initialisation_date),
            len(forecast_horizon),
            len(lat),
            len(lon),
        )
    )

    ds = xr.Dataset(
        {
            "precip": (
                ["number", "initialisation_date", "forecast_horizon", "lat", "lon"],
                precip,
            )
        },
        coords={
            "lon": lon,
            "lat": lat,
            "initialisation_date": initialisation_date,
            "number": number,
            "forecast_horizon": forecast_horizon,
            "valid_time": (["initialisation_date", "step"], valid_time),
        },
    )
    return ds


def make_dummy_ouce_s5_data(tmp_path: Path) -> Path:
    number = [i for i in range(0, 2)]  # corresponds to ensemble number (26)
    latitude = np.linspace(0, 180, 180)
    longitude = np.linspace(0, 360, 360)
    time = pd.date_range("2008-02-01", periods=10, freq="6H")

    t2m = np.ones(shape=(len(time), len(number), len(latitude), len(longitude)))
    coords = {
        "longitude": longitude,
        "latitude": latitude,
        "time": time,
        "number": number,
    }
    dims = ["time", "number", "latitude", "longitude"]
    ds = xr.Dataset({"t2m": (dims, t2m)}, coords=coords)

    out_dir = Path("seas5/1.0x1.0/6-hourly")
    out_dir = out_dir / "2m_temperature/nc"
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / "seas5_6-hourly_2m_temperature_200802.nc"
    ds.to_netcdf(path)

    return path


def save_dummy_seas5(
    tmp_path,
    date_str,
    to_grib=False,
    dataset="seasonal-monthly-pressure-levels",
    variable="temperature",
) -> Path:
    """
    filename structure:
     data/raw/seasonal-monthly-pressure-levels/temperature/2017/01.grib
    """
    year = pd.to_datetime(date_str).year
    month = pd.to_datetime(date_str).month
    out_dir = tmp_path / "data" / "raw" / dataset / variable / str(year)
    if not out_dir.exists():
        out_dir.mkdir(exist_ok=True, parents=True)

    ds = make_dummy_seas5_data(date_str)
    if to_grib:
        cfgrib.to_grib(
            ds,
            out_dir / f"{month:02}.grib",
            grib_keys={"edition": 2, "gridType": "regular_ll"},
        )
    else:
        ds.to_netcdf(out_dir / f"{month:02}.nc")

    return out_dir


class TestS5Preprocessor:
    def test_initialisation(self, tmp_path):
        data_dir = tmp_path / "data"
        if not data_dir.exists():
            data_dir.mkdir(exist_ok=True, parents=True)

        S5Preprocessor(data_dir)
        assert (data_dir / "interim" / "s5_preprocessed").exists()
        assert (data_dir / "interim" / "s5_interim").exists()

    def test_preprocess_ouce_data(self, tmp_path):
        ouce_data_path = make_dummy_ouce_s5_data(tmp_path)
        o = OuceS5Data()
        ds = o.read_ouce_s5_data(ouce_data_path, infer=True)
        assert "forecast_horizon" in [v for v in ds.coords]
        assert "initialisation_date" in [v for v in ds.coords]
        assert len(ds.time.values.shape), "Expect a 2D `time` coordinate"

    def test_find_grib_file(self, tmp_path):
        # create grib file to test if it can be found by s5 preprocessor
        _ = save_dummy_seas5(
            tmp_path,
            "2018-01-01",
            to_grib=True,
            dataset="seasonal-monthly-pressure-levels",
            variable="temperature",
        )
        out_dir = tmp_path / "data" / "raw" / "seasonal-monthly-pressure-levels"
        out_dir = out_dir / "temperature" / "2018"
        assert (out_dir / "01.grib").exists()

        processor = S5Preprocessor(tmp_path / "data")

        # check the preprocessor can find the grib file created
        fpaths = processor.get_filepaths(
            grib=True, target_folder=processor.raw_folder, variable="temperature"
        )
        assert fpaths[0].name == "01.grib", (
            f"unable to find the created dataset"
            "at data/raw/s5/seasonal-monthly-pressure-levels"
        )

    def test_preprocess(self, tmp_path):
        out_dir = tmp_path / "data" / "raw" / "s5"
        out_dir = (
            out_dir / "seasonal-monthly-pressure-levels" / "2m_temperature" / str(2018)
        )
        if not out_dir.exists():
            out_dir.mkdir(exist_ok=True, parents=True)

        # preprocessor working with pretend ouce data (because writing to .grib is failing)
        ouce_dir = make_dummy_ouce_s5_data(tmp_path)
        kenya = get_kenya()
        regrid_dataset, _, _ = _make_dataset(
            size=(20, 20),
            latmin=kenya.latmin,
            latmax=kenya.latmax,
            lonmin=kenya.lonmin,
            lonmax=kenya.lonmax,
        )

        # the reference dataset to regrid to
        regrid_path = tmp_path / "regridder.nc"
        regrid_dataset.to_netcdf(regrid_path)

        # run the preprocessing
        processor = S5Preprocessor(tmp_path / "data", ouce_server=True)

        processor.preprocess(
            subset_str="kenya",
            regrid=regrid_path,
            variable="2m_temperature",
            cleanup=True,
            **dict(ouce_dir=ouce_dir.parents[2], infer=True),
        )

        # check preprocessed file exists
        assert (
            processor.preprocessed_folder / "s5_preprocessed" / "s5_t2m_kenya.nc"
        ).exists(), (
            "Expecting to find the kenyan_subset netcdf file"
            "at the preprocessed / s5_preprocessed / s5_{variable}_{subset_str}.nc"
        )

        # open the data
        out_data = xr.open_dataset(
            processor.preprocessed_folder / "s5_preprocessed" / "s5_t2m_kenya.nc"
        )

        # check the subsetting happened properly
        expected_dims = [
            "lat",
            "lon",
            "initialisation_date",
            "forecast_horizon",
            "number",
        ]
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

        # check the lat/lon is the correct shape
        assert out_data.t2m.values.shape[-2:] == (20, 20)

        # test the stacking to select the forecast time
        # NOTE: this is how you select data from the S5 data for the `real time`
        out_data["valid_time"] = (
            out_data.initialisation_date + out_data.forecast_horizon
        )
        stacked = out_data.stack(time=("initialisation_date", "forecast_horizon"))
        assert stacked.time.shape == (10,), "should be a 1D vector"
        selected = stacked.swap_dims({"time": "valid_time"}).sel(valid_time="2008-03")

        assert selected.time.size == 6, (
            "Should have only selected 6 timesteps"
            " for the month 2008-03. The calculation of valid_time is "
            "complicated but it should select the forecasts that enter into"
            "the month of interest."
        )

        # check the cleanup has worked
        assert (
            not processor.interim.exists()
        ), f"Interim S5 folder should have been deleted"
