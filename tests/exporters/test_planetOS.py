import boto3
from moto import mock_s3

from itertools import product


from src.exporters import ERA5ExporterPOS


class TestERA5ExporterPOS:
    @mock_s3
    def test_available_years(self, tmp_path):

        # setup our fake bucket
        era5_bucket = "era5-pds"
        conn = boto3.client("s3")
        conn.create_bucket(Bucket=era5_bucket)

        files = [f"{year}/main.nc" for year in range(2008, 2019)]

        # This will check we don't erroneously add non-year folders
        files.append("QA/main.nc")

        for file in files:
            conn.put_object(Bucket=era5_bucket, Key=file, Body="")

        exporter = ERA5ExporterPOS(tmp_path)
        years = exporter._get_available_years()

        assert years == list(
            range(2008, 2019)
        ), f"Expected exporter to retrieve {list(range(2008, 2019))} years, got {years} instead"

    @mock_s3
    def test_available_variables(self, tmp_path):

        # setup our fake bucket
        era5_bucket = "era5-pds"
        conn = boto3.client("s3")
        conn.create_bucket(Bucket=era5_bucket, ACL="public-read")

        expected_variables = {"a", "b", "c", "d", "e"}

        for variable in expected_variables:
            key = f"2008/01/data/{variable}.nc"
            conn.put_object(Bucket=era5_bucket, Key=key, Body="")

        exporter = ERA5ExporterPOS(tmp_path)
        returned_variables = exporter.get_variables(2008, 1)

        assert len(returned_variables) == len(
            expected_variables
        ), f"Expected {len(expected_variables)} to be returned, got {len(returned_variables)}"

        for variable in expected_variables:
            assert (
                variable in returned_variables
            ), f"Expected to get variable {variable} but did not"

    @mock_s3
    def test_export(self, tmp_path):
        # setup our fake bucket
        era5_bucket = "era5-pds"
        conn = boto3.client("s3")
        conn.create_bucket(Bucket=era5_bucket, ACL="public-read")

        variable = "precipitation"
        years = range(2008, 2019)
        months = range(1, 12 + 1)

        keys = []
        expected_files = []
        for year, month in product(years, months):
            key = f"{year}/{month:02d}/data/{variable}.nc"
            keys.append(key)

            filename = tmp_path / f"raw/era5POS/{year}/{month:02d}/{variable}.nc"
            expected_files.append(filename)

        # This will check we don't erroneously add non-year folders
        keys.append("QA/main.nc")

        for key in keys:
            conn.put_object(Bucket=era5_bucket, Key=key, Body="")

        exporter = ERA5ExporterPOS(tmp_path)
        downloaded_files = exporter.export(variable)

        for file in expected_files:
            assert file.exists(), f"Expected {file} to be downloaded"

        assert len(expected_files) == len(downloaded_files), (
            f"Expected {len(expected_files)} files to be downloaded, "
            f"got {len(downloaded_files)} instead"
        )

        for file in expected_files:
            assert (
                file in downloaded_files
            ), f"{file} not returned by the export function"
