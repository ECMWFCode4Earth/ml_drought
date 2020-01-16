from unittest.mock import patch, Mock
import numpy as np
import pytest

from src.exporters.era5_land import ERA5LandExporter


class TestERA5LandExporter:
    @pytest.mark.xfail(reason="cdsapi may not be installed")
    @patch("cdsapi.Client")
    def test_initialisation(self, cdsapi_mock, tmp_path):
        cdsapi_mock.return_value = Mock()

        exporter = ERA5LandExporter(tmp_path)

        assert exporter.dataset == "reanalysis-era5-land"
        assert exporter.granularity == "hourly"

    @pytest.mark.xfail(reason="cdsapi may not be installed")
    @patch("cdsapi.Client")
    def test_make_filename(self, cdsapi_mock, tmp_path):
        cdsapi_mock.return_value = Mock()

        exporter = ERA5LandExporter(tmp_path)
        selection_request = exporter.create_selection_request(
            variable="snowmelt",
            selection_request=None,
            granularity=exporter.granularity,
        )
        fname = exporter.make_filename(exporter.dataset, selection_request)

        # data/raw/{dataset}/{variable}/{year(s)}/{month(s)}
        expected = "raw/reanalysis-era5-land/snowmelt/2001_2019/01_12.nc"

        assert expected in fname.as_posix(), (
            "Wrong filename created!" f"Got: {fname.as_posix()} \nExpected: {expected}"
        )

    @pytest.mark.xfail(reason="cdsapi may not be installed")
    @patch("cdsapi.Client")
    def test_selection_request(self, cdsapi_mock, tmp_path):
        cdsapi_mock.return_value = Mock()

        exporter = ERA5LandExporter(tmp_path)
        selection_request = exporter.create_selection_request(
            variable="snowmelt",
            selection_request=None,
            granularity=exporter.granularity,
        )

        expected_keys = ["format", "variable", "year", "month", "time", "day", "area"]
        assert np.isin(list(selection_request.keys()), expected_keys).all(), (
            "" f"Expected: {expected_keys} \nGot: {list(selection_request.keys())}"
        )

        assert selection_request["variable"][0] == "snowmelt"

    @pytest.mark.xfail(reason="cdsapi may not be installed")
    @patch("cdsapi.Client")
    def test_wrong_variable(self, cdsapi_mock, tmp_path):
        cdsapi_mock.return_value = Mock()

        exporter = ERA5LandExporter(tmp_path)
        with pytest.raises(AssertionError) as err:
            exporter.export("hello im not a real variable")
            err.match(r"Need to select a variable*")

    @pytest.mark.xfail(reason="cdsapi may not be installed")
    @patch("cdsapi.Client")
    def test_default_selection_request(self, cdsapi_mock, tmp_path):
        cdsapi_mock.return_value = Mock()

        exporter = ERA5LandExporter(tmp_path)
        default_selection_request = exporter.create_selection_request("snowmelt")
        expected_selection_request = {
            "format": "netcdf",
            "variable": ["snowmelt"],
            "year": [
                "2001",
                "2002",
                "2003",
                "2004",
                "2005",
                "2006",
                "2007",
                "2008",
                "2009",
                "2010",
                "2011",
                "2012",
                "2013",
                "2014",
                "2015",
                "2016",
                "2017",
                "2018",
                "2019",
            ],
            "month": [
                "01",
                "02",
                "03",
                "04",
                "05",
                "06",
                "07",
                "08",
                "09",
                "10",
                "11",
                "12",
            ],
            "time": [
                "00:00",
                "01:00",
                "02:00",
                "03:00",
                "04:00",
                "05:00",
                "06:00",
                "07:00",
                "08:00",
                "09:00",
                "10:00",
                "11:00",
                "12:00",
                "13:00",
                "14:00",
                "15:00",
                "16:00",
                "17:00",
                "18:00",
                "19:00",
                "20:00",
                "21:00",
                "22:00",
                "23:00",
            ],
            "day": [
                "01",
                "02",
                "03",
                "04",
                "05",
                "06",
                "07",
                "08",
                "09",
                "10",
                "11",
                "12",
                "13",
                "14",
                "15",
                "16",
                "17",
                "18",
                "19",
                "20",
                "21",
                "22",
                "23",
                "24",
                "25",
                "26",
                "27",
                "28",
                "29",
                "30",
                "31",
            ],
            "area": "6.002/33.501/-5.202/42.283",
        }
        for key, val in expected_selection_request.items():
            default_val = default_selection_request[key]
            assert default_val == val, f"For {key}, expected {val}, got {default_val}"

    @pytest.mark.xfail(reason="cdsapi may not be installed")
    @patch("cdsapi.Client")
    def test_user_defined_selection_requests(self, cdsapi_mock, tmp_path):
        cdsapi_mock.return_value = Mock()
        exporter = ERA5LandExporter(tmp_path)

        user_defined_arguments = {"year": [2019], "day": [1], "month": [1], "time": [0]}
        default_selection_request = exporter.create_selection_request(
            "snowmelt", user_defined_arguments
        )
        expected_selection_request = {
            "format": "netcdf",
            "variable": ["snowmelt"],
            "year": ["2019"],
            "month": ["01"],
            "time": ["00:00"],
            "day": ["01"],
            "area": "6.002/33.501/-5.202/42.283",
        }
        for key, val in expected_selection_request.items():
            default_val = default_selection_request[key]
            assert default_val == val, f"For {key}, expected {val}, got {default_val}"

    @pytest.mark.xfail(reason="cdsapi may not be installed")
    @patch("cdsapi.Client")
    def test_break_up(self, cdsapi_mock, tmp_path):
        cdsapi_mock.return_value = Mock()
        exporter = ERA5LandExporter(tmp_path)

        user_defined_arguments = {
            "year": [2019, 2018],
            "day": [1, 2],
            "month": [4, 5],
            "time": [0],
        }

        # MONTHLY
        output_paths = exporter.export(
            "snowmelt",
            selection_request=user_defined_arguments,
            break_up="monthly",
            n_parallel_requests=1,
        )

        raw_folder = tmp_path / "raw"
        expected_paths = [
            raw_folder / "reanalysis-era5-land/snowmelt/2018/04.nc",
            raw_folder / "reanalysis-era5-land/snowmelt/2018/05.nc",
            raw_folder / "reanalysis-era5-land/snowmelt/2019/04.nc",
            raw_folder / "reanalysis-era5-land/snowmelt/2019/05.nc",
        ]

        assert len(output_paths) == len(expected_paths), (
            f"Expected {len(expected_paths)} " f"files, got {len(output_paths)}"
        )

        for file in expected_paths:
            assert file in output_paths, f"{file} not in the output paths!"

        # YEARLY
        output_paths = exporter.export(
            "snowmelt",
            selection_request=user_defined_arguments,
            break_up="yearly",
            n_parallel_requests=1,
        )

        raw_folder = tmp_path / "raw"
        expected_paths = [
            raw_folder / "reanalysis-era5-land/snowmelt/2018/04_05.nc",
            raw_folder / "reanalysis-era5-land/snowmelt/2019/04_05.nc",
        ]

        assert len(output_paths) == len(expected_paths), (
            f"Expected {len(expected_paths)} " f"files, got {len(output_paths)}"
        )

        for file in expected_paths:
            assert file in output_paths, f"{file} not in the output paths!"
