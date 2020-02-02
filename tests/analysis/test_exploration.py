import xarray as xr
import numpy as np
from pandas._libs.tslibs.timestamps import Timestamp

from src.analysis.exploration import (
    calculate_seasonal_anomalies,
    calculate_seasonal_anomalies_spatial,
    create_anomaly_df,
)

from ..utils import _create_dummy_precip_data


class TestExploration:
    def test_seasonal_anomalies(self, tmp_path):
        _create_dummy_precip_data(tmp_path)
        ds = xr.open_dataset(
            tmp_path / "data" / "interim" / "chirps_preprocessed" / "data_kenya.nc"
        )

        da = calculate_seasonal_anomalies(ds, "precip")
        assert da.shape == (12,)
        assert np.isin(["time", "season"], da.coords).all()
        assert da.name == "anomaly"

    def test_calculate_seasonal_anomalies_spatial(self, tmp_path):
        _create_dummy_precip_data(tmp_path)
        ds = xr.open_dataset(
            tmp_path / "data" / "interim" / "chirps_preprocessed" / "data_kenya.nc"
        )
        da = calculate_seasonal_anomalies_spatial(ds, "precip")
        assert da.name == "anomaly"
        assert da.shape == (12, 30, 30)
        n_seasons = len(ds.resample(time="Q-DEC").mean().time)
        assert len(da.time) == n_seasons

    def test_create_anomaly_df(self, tmp_path):
        _create_dummy_precip_data(tmp_path)
        ds = xr.open_dataset(
            tmp_path / "data" / "interim" / "chirps_preprocessed" / "data_kenya.nc"
        )
        da = calculate_seasonal_anomalies(ds, "precip")
        df = create_anomaly_df(da)

        assert isinstance(df.time[0], Timestamp)
        assert np.isin(["time", "anomaly"], df.columns).all()
