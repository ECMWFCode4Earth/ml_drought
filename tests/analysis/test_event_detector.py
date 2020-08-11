import numpy as np
import pandas as pd
import xarray as xr
from src.analysis import EventDetector

from ..utils import _create_dummy_precip_data


class TestEventDetector:
    @staticmethod
    def create_test_consec_data(tmp_path):
        """
        create an array of 20 timesteps with 2 events below 1STD of the series
        event from indices 3:7 & 10:13
        """
        a = np.ones((400)) * 10
        a[3:7] = 0.2
        a[10:13] = 0.2
        p = np.repeat(a, 25).reshape(400, 5, 5)

        lat = np.arange(0, 5)
        lon = np.arange(0, 5)
        time = pd.date_range("2000-01-01", freq="M", periods=p.shape[0])

        d = xr.Dataset(
            {"precip": (["time", "lat", "lon"], p)},
            coords={"lon": lon, "lat": lat, "time": time},
        )

        data_dir = tmp_path / "data" / "interim" / "chirps_preprocessed"
        if not data_dir.exists():
            data_dir.mkdir(parents=True, exist_ok=True)

        out_path = data_dir / "data_kenya.nc"
        d.to_netcdf(out_path)
        return out_path

    def test_initialise_event_detector(self, tmp_path):
        _create_dummy_precip_data(tmp_path)

        data_dir = tmp_path / "data" / "interim"
        precip_dir = data_dir / "chirps_preprocessed" / "data_kenya.nc"

        e = EventDetector(precip_dir)

        assert "precip" in [
            v for v in e.ds.variables
        ], f"\
        Expected precip to be a variable.\
        Got: {[v for v in e.ds.variables]}"

        assert e.ds.precip.shape == (
            36,
            30,
            30,
        ), f"\
        Expected shape of EventDetector.ds to be {(36, 30, 30)}.\
        Got: {e.ds.precip.shape}"

    def test_number_of_consecutive_events(self, tmp_path):
        in_path = self.create_test_consec_data(tmp_path)

        e = EventDetector(in_path)
        e.detect(variable="precip", time_period="dayofyear", hilo="low", method="std")

        # check boolean dtype
        assert e.exceedences.dtype == np.dtype(
            "bool"
        ), f"Expected a\
        boolean array but got: {e.exceedences.dtype}"

        # check the threshold calculation
        assert (
            e.thresh.precip.max().values == 10
        ), f"Expected max to be 10\
        got: {e.thresh.precip.max().values}"

        # test run size
        runs = e.calculate_runs()
        assert (
            runs.max().values == 4
        ), f"Expected max run length to be 4\
        Got: {runs.max().values}"

        # ensure they are of the correct time slice
        max_run_time = runs.where(runs == runs.max(), drop=True).squeeze()
        expected = pd.to_datetime("2000-04-30")
        got = pd.to_datetime(max_run_time.time.values)
        assert (
            got == expected
        ), f"Expected\
        the maximum time slice to be: {expected} Got: {got}"

        # ensure the second largest timeslice is
        second_largest_run_time = runs.where(runs == 3, drop=True).squeeze()
        expected = pd.to_datetime("2000-11-30")
        got = pd.to_datetime(second_largest_run_time.time.values)
        assert (
            got == expected
        ), f"Expected\
        the second largest time slice to be: {expected} Got: {got}"

    def test_q90(self, tmp_path):
        # TEST the mean threshold
        in_path = self.create_test_consec_data(tmp_path)

        e = EventDetector(in_path)
        # ABOVE Q90
        e.detect(variable="precip", time_period="month", hilo="high", method="q90")

        assert (
            len(e.thresh.month) == 12
        ), f"Expected the threshold \
        calculation to be 12 (one for each month)"

        # all monthly thresholds should be 10
        got = e.thresh.precip.mean(dim=["lat", "lon"]).values
        assert all(
            got == 10
        ), f"\
        Expected all monthly threshold values in e.thresh to be 10. Got: {got}"

        # longest run should be 0 (exceeding Q90)
        runs = e.calculate_runs()
        assert (
            runs.max().values == 0
        ), f"Expected the longest run to be 0 (exceeding q90)"

    def test_q10(self, tmp_path):
        # TEST the q10 threshold
        in_path = self.create_test_consec_data(tmp_path)

        e = EventDetector(in_path)
        # BELOW Q10
        e.detect(variable="precip", time_period="month", hilo="low", method="q10")

        assert (
            len(e.thresh.month) == 12
        ), f"Expected the threshold \
        calculation to be 12 (one for each month)"

        # all monthly thresholds should be 10
        got = e.thresh.precip.mean(dim=["lat", "lon"]).values
        assert all(
            got == 10
        ), f"\
        Expected all monthly threshold values in e.thresh to be 10. Got: {got}"

        # longest run should be 0 (below Q10)
        runs = e.calculate_runs()
        assert runs.max().values == 4, f"Expected the longest run to be 4 (below q10)"

    def test_abs(self, tmp_path):
        # TEST the absolute threshold
        in_path = self.create_test_consec_data(tmp_path)

        e = EventDetector(in_path)
        # BELOW Q10
        e.detect(
            variable="precip", time_period="month", hilo="low", method="abs", value=0.3
        )

        assert (
            len(e.thresh.month) == 12
        ), f"Expected the threshold \
        calculation to be 12 (one for each month)"

        # all monthly thresholds should be 10
        got = e.thresh.precip.mean(dim=["lat", "lon"]).values
        assert all(
            got == 0.3
        ), f"\
        Expected all monthly threshold values in e.thresh to be 0.3. Got: {got}"

        # longest run should be 4 (below 0.3)
        runs = e.calculate_runs()
        assert runs.max().values == 4, f"Expected the longest run to be 4 (below 0.3)"
