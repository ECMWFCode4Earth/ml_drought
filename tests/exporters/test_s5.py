from unittest.mock import patch, Mock
import pytest
import numpy as np

from src.exporters import S5Exporter
from src.exporters.seas5.all_valid_s5 import datasets as dataset_reference


class TestS5Exporter:
    @pytest.mark.xfail(reason="cdsapi may not be installed")
    @patch("cdsapi.Client")
    def test_failure_on_invalid_granularity(self, tmp_path):
        s5 = S5Exporter(pressure_level=True, data_folder=tmp_path)

        with pytest.raises(AssertionError) as e:
            s5.get_s5_initialisation_times("daily")
            e.match(r"Invalid granularity*")

    @pytest.mark.xfail(reason="cdsapi may not be installed")
    @patch("cdsapi.Client")
    def test_initialisation_times_produces_correct_keys(self, tmp_path):
        s5 = S5Exporter(pressure_level=True, data_folder=tmp_path)
        selection_request = s5.get_s5_initialisation_times(
            "hourly", min_year=2017, max_year=2018, min_month=1, max_month=1
        )

        expected_keys = ["year", "month", "day"]
        assert all(
            np.isin(expected_keys, [k for k in selection_request.keys()])
        ), f"Expecting keys: {expected_keys}. \
        Got: {[k for k in selection_request.keys()]}"

    @pytest.mark.xfail(reason="cdsapi may not be installed")
    @patch("cdsapi.Client")
    def test_leadtimes_produces_correct_keys_values_hourly(self, tmp_path):

        granularity = "hourly"
        pressure_level = False

        s5 = S5Exporter(
            data_folder=tmp_path, granularity=granularity, pressure_level=pressure_level
        )

        # 5 days because hourly data
        max_leadtime = 5
        selection_request = s5.get_s5_leadtimes(max_leadtime)

        # assert returning correct leadtime
        expected_keys = ["leadtime_hour"]
        assert all(
            np.isin(expected_keys, [k for k in selection_request.keys()])
        ), f"Expecting keys: {expected_keys}. \
        Got: {[k for k in selection_request.keys()]}"

        # assert that returning a list of strings
        assert isinstance(
            selection_request["leadtime_hour"][0], str
        ), f"Expected a list of str, got: {type(selection_request['leadtime_hour'][0])}"

        # assert that returning the correct max forecast hour
        leadtimes_int = [int(lt) for lt in selection_request["leadtime_hour"]]
        assert (
            max(leadtimes_int) == 120
        ), f"Expected max leadtime to be 120hrs\
        Got: {max(leadtimes_int)}hrs"

    @pytest.mark.xfail(reason="cdsapi may not be installed")
    @patch("cdsapi.Client")
    def test_leadtimes_produces_correct_keys_values_monthly(self, tmp_path):
        granularity = "monthly"
        pressure_level = False

        s5 = S5Exporter(
            data_folder=tmp_path, granularity=granularity, pressure_level=pressure_level
        )

        # 5 months because monthly data
        max_leadtime = 5
        selection_request = s5.get_s5_leadtimes(max_leadtime)

        # assert returning correct keys
        expected_keys = ["leadtime_month"]
        assert all(
            np.isin(expected_keys, [k for k in selection_request.keys()])
        ), f"Expecting keys: {expected_keys}. \
        Got: {[k for k in selection_request.keys()]}"

        # assert that returning a list of strings
        assert isinstance(
            selection_request["leadtime_month"][0], str
        ), f"Expected a list of str, got: {type(selection_request['leadtime_month'][0])}"

        # assert that returning the correct max forecast month
        leadtimes_int = [int(lt) for lt in selection_request["leadtime_month"]]
        assert (
            max(leadtimes_int) == 5
        ), f"Expected max leadtime to be 5months\
        Got: {max(leadtimes_int)}months"

    @pytest.mark.xfail(reason="cdsapi may not be installed")
    @patch("cdsapi.Client")
    def test_dataset_reference_creation(self, tmp_path):
        granularity = "monthly"
        pressure_level = False

        s5 = S5Exporter(
            data_folder=tmp_path, granularity=granularity, pressure_level=pressure_level
        )

        expected_dataset_reference = dataset_reference["seasonal-monthly-single-levels"]

        assert (
            s5.dataset_reference == expected_dataset_reference
        ), f"for dataset seasonal-monthly-single-levels we are not \
        getting the expected_dataset_reference"

    @pytest.mark.xfail(reason="cdsapi may not be installed")
    @patch("cdsapi.Client")
    def test_None_product_type_for_original_single_levels(self, tmp_path):
        granularity = "hourly"
        pressure_level = False
        s5 = S5Exporter(
            data_folder=tmp_path, granularity=granularity, pressure_level=pressure_level
        )
        assert (
            s5.get_product_type() is None
        ), f"Expected the product_type for\
        seasonal-original-single-levels dataset to be None. Got:\
        {s5.get_product_type()}"

    @pytest.mark.xfail(reason="cdsapi may not be installed")
    @patch("cdsapi.Client")
    def test_product_type_single_levels_monthly(self, tmp_path):
        granularity = "monthly"
        pressure_level = False
        s5 = S5Exporter(
            data_folder=tmp_path, granularity=granularity, pressure_level=pressure_level
        )

        # monthly_mean is the product_type
        assert (
            s5.get_product_type() == "monthly_mean"
        ), f"\
        Expecting `product_type` for `seasonal-original-single-levels` \
        dataset to be 'monthly_mean'. Returned: {s5.get_product_type(None)}"

        # hindcast_climate_mean is a valid product type
        assert (
            s5.get_product_type("hindcast_climate_mean") == "hindcast_climate_mean"
        ), f"\
        Expecting `product_type` for `seasonal-original-single-levels`\
        dataset to be 'hindcast_climate_mean' Got: {s5.get_product_type('hindcast_climate_mean')}"

        # assert erroneous product_type
        with pytest.raises(AssertionError) as e:
            s5.get_product_type("dsgdfgdfh")
            e.match(r"Invalid `product_type`*")

    @pytest.mark.xfail(reason="cdsapi may not be installed")
    @patch("cdsapi.Client")
    def test_create_selection_request(self, tmp_path):
        granularity = "monthly"
        pressure_level = False

        s5 = S5Exporter(
            data_folder=tmp_path, granularity=granularity, pressure_level=pressure_level
        )

        variable = "total_precipitation"
        max_leadtime = 5
        min_year = 2017
        max_year = 2018
        min_month = 1
        max_month = 12

        processed_selection_request = s5.create_selection_request(
            variable=variable,
            max_leadtime=max_leadtime,
            min_year=min_year,
            max_year=max_year,
            min_month=min_month,
            max_month=max_month,
        )

        # CHECK default arguments
        assert (
            processed_selection_request["originating_centre"] == "ecmwf"
        ), "\
        Expected originating_centre to be: {'ecmwf'}. Got:\
        {processed_selection_request['originating_centre']}"

        assert (
            processed_selection_request["system"] == "5"
        ), f"\
        Expected 'system' to be '5'. Got:\
        {processed_selection_request['system']}"

        # CHECK time arguments
        assert processed_selection_request["year"] == [
            "2017",
            "2018",
        ], f"\
        Expected 'year' to be ['2017','2018']. Got:\
        {processed_selection_request['year']}"

        exp_months = [
            "{:02d}".format(month) for month in range(min_month, max_month + 1)
        ]
        assert (
            processed_selection_request["month"] == exp_months
        ), f"\
        Expected 'month' to be {exp_months}. Got:\
        {processed_selection_request['month']}"

    @pytest.mark.xfail(reason="cdsapi may not be installed")
    @patch("cdsapi.Client")
    def test_expected_filepath(self, cdsapi_mock, tmp_path):
        cdsapi_mock.return_value = Mock()
        granularity = "monthly"
        pressure_level = False

        s5 = S5Exporter(
            data_folder=tmp_path, granularity=granularity, pressure_level=pressure_level
        )

        expected_filepath = (
            tmp_path
            / "raw/seasonal-monthly-single-levels/\
            total_precipitation/2017_2018/Y2017_2018_M01_12.grib"
        ).as_posix()
        expected_filepath = expected_filepath.replace(" ", "")

        variable = "total_precipitation"
        max_leadtime = 5
        min_year = 2017
        max_year = 2018
        min_month = 1
        max_month = 12

        processed_selection_request = s5.create_selection_request(
            variable=variable,
            max_leadtime=max_leadtime,
            min_year=min_year,
            max_year=max_year,
            min_month=min_month,
            max_month=max_month,
        )

        filepath = s5.make_filename(
            dataset=s5.dataset, selection_request=processed_selection_request
        )

        assert (
            expected_filepath == filepath.as_posix()
        ), f"\
            Expected: {expected_filepath}. Got: {filepath.as_posix()}"

    @pytest.mark.xfail(reason="cdsapi may not be installed")
    @patch("cdsapi.Client")
    def test_dataset_created_properly(self, tmp_path):

        expected_datasets = [
            "seasonal-monthly-single-levels",
            "seasonal-monthly-pressure-levels",
            "seasonal-original-single-levels",
            "seasonal-original-pressure-levels",
        ]
        pressure_levels = [False, True, False, True]
        granularities = ["monthly", "monthly", "hourly", "hourly"]

        for ix, expected in enumerate(expected_datasets):
            s5 = S5Exporter(
                data_folder=tmp_path,
                granularity=granularities[ix],
                pressure_level=pressure_levels[ix],
            )

            assert (
                s5.dataset == expected
            ), f"\
            Expected: {expected}. Got: {s5.dataset}"

    @pytest.mark.xfail(reason="cdsapi may not be installed")
    @patch("cdsapi.Client")
    def test_export_functionality(self, cdsapi_mock, tmp_path):
        cdsapi_mock.return_value = Mock()
        granularity = "monthly"
        pressure_level = False

        s5 = S5Exporter(
            data_folder=tmp_path, granularity=granularity, pressure_level=pressure_level
        )

        variable = "total_precipitation"
        min_year = 2017
        max_year = 2017
        min_month = 1
        max_month = 1
        max_leadtime = 1
        n_parallel_requests = 1
        show_api_request = True

        s5.export(
            variable=variable,
            min_year=min_year,
            max_year=max_year,
            min_month=min_month,
            max_month=max_month,
            max_leadtime=max_leadtime,
            show_api_request=show_api_request,
            n_parallel_requests=n_parallel_requests,
        )

        (
            tmp_path
            / "raw/seasonal-monthly-single-levels\
            /total_precipitation/2017/M01.grib"
        ).as_posix().replace(" ", "")
        cdsapi_mock.assert_called()
