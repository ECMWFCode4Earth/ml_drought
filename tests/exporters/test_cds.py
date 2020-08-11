from pathlib import Path
import pytest
from unittest.mock import patch, Mock

from src.exporters.cds import CDSExporter, ERA5Exporter
from src.exporters.base import get_kenya


class TestCDSExporter:
    @pytest.mark.xfail(reason="cdsapi may not be installed")
    @patch("cdsapi.Client")
    def test_filename_multiple(self, cdsapi_mock, tmp_path):
        cdsapi_mock.return_value = Mock()
        exporter = CDSExporter(tmp_path)

        dataset = "megadodo-publications"
        selection_request = {
            "variable": ["towel"],
            "year": [1979, 1978, 1980],
            "month": [10, 11, 12],
        }

        filename = exporter.make_filename(dataset, selection_request)
        # first, we check the filename is right
        constructed_filepath = Path(
            "raw/megadodo-publications/towel/1978_1980/10_12.nc"
        )
        expected = tmp_path / constructed_filepath
        assert filename == expected, f"Got {filename}, expected {expected}!"

        # then, we check all the files were correctly made
        assert expected.parents[0].exists(), "Folders not correctly made!"

    @pytest.mark.xfail(reason="cdsapi may not be installed")
    @patch("cdsapi.Client")
    def test_filename_single(self, cdsapi_mock, tmp_path):
        cdsapi_mock.return_value = Mock()
        exporter = CDSExporter(tmp_path)

        dataset = "megadodo-publications"
        selection_request = {"variable": ["towel"], "year": [1979], "month": [10]}

        filename = exporter.make_filename(dataset, selection_request)
        # first, we check the filename is right
        constructed_filepath = Path("raw/megadodo-publications/towel/1979/10.nc")
        expected = tmp_path / constructed_filepath
        assert filename == expected, f"Got {filename}, expected {expected}!"

        # then, we check all the files were correctly made
        assert expected.parents[0].exists(), "Folders not correctly made!"

    def test_selection_dict_granularity(self):

        selection_dict_monthly = ERA5Exporter.get_era5_times(granularity="monthly")
        assert (
            "day" not in selection_dict_monthly
        ), "Got day values in monthly the selection dict!"

        selection_dict_hourly = ERA5Exporter.get_era5_times(granularity="hourly")
        assert (
            "day" in selection_dict_hourly
        ), "Day values not in hourly selection dict!"

    def test_area(self):

        region = get_kenya()
        kenya_str = CDSExporter.create_area(region)

        expected_str = "6.002/33.501/-5.202/42.283"
        assert kenya_str == expected_str, f"Got {kenya_str}, expected {expected_str}!"

    @pytest.mark.xfail(reason="cdsapi may not be installed")
    @patch("cdsapi.Client")
    def test_default_selection_request(self, cdsapi_mock, tmp_path):
        cdsapi_mock.return_value = Mock()
        exporter = ERA5Exporter(tmp_path)
        default_selection_request = exporter.create_selection_request("precipitation")
        expected_selection_request = {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": ["precipitation"],
            "year": [
                "1979",
                "1980",
                "1981",
                "1982",
                "1983",
                "1984",
                "1985",
                "1986",
                "1987",
                "1988",
                "1989",
                "1990",
                "1991",
                "1992",
                "1993",
                "1994",
                "1995",
                "1996",
                "1997",
                "1998",
                "1999",
                "2000",
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
        exporter = ERA5Exporter(tmp_path)

        user_defined_arguments = {"year": [2019], "day": [1], "month": [1], "time": [0]}
        default_selection_request = exporter.create_selection_request(
            "precipitation", user_defined_arguments
        )
        expected_selection_request = {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": ["precipitation"],
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
        exporter = ERA5Exporter(tmp_path)

        user_defined_arguments = {
            "year": [2019, 2018],
            "day": [1, 2],
            "month": [4, 5],
            "time": [0],
        }

        output_paths = exporter.export(
            "precipitation",
            dataset="era5",
            granularity="hourly",
            selection_request=user_defined_arguments,
            break_up=True,
            n_parallel_requests=1,
        )

        raw_folder = tmp_path / "raw"
        expected_paths = [
            raw_folder / "era5/precipitation/2018/04_05.nc",
            raw_folder / "era5/precipitation/2019/04_05.nc",
        ]

        assert len(output_paths) == len(expected_paths), (
            f"Expected {len(expected_paths)} " f"files, got {len(output_paths)}"
        )

        for file in expected_paths:
            assert file in output_paths, f"{file} not in the output paths!"

    def test_correct_inputs(self):

        user_defined_arguments = {"year": 2019, "day": 1, "month": 5, "time": "00:00"}

        expected_iterable = {
            "year": [2019],
            "day": [1],
            "month": [5],
            "time": ["00:00"],
        }

        expected_input = {
            "year": ["2019"],
            "day": ["01"],
            "month": ["05"],
            "time": ["00:00"],
        }

        for key, val in user_defined_arguments.items():
            corrected_iter = ERA5Exporter._check_iterable(val, key)
            assert (
                corrected_iter == expected_iterable[key]
            ), f"When checking iterable, expected {expected_iterable[key]}, got {corrected_iter}"
            corrected_input = [
                ERA5Exporter._correct_input(x, key) for x in corrected_iter
            ]
            assert (
                corrected_input == expected_input[key]
            ), f"When checking iterable, expected {expected_input[key]}, got {corrected_input}"
