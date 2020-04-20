import urllib.request
from unittest.mock import patch, MagicMock
import os
from pathlib import Path
import pytest

from src.exporters.boku_ndvi import BokuNDVIExporter

"""
TODO:
- Test both 250m and 1000m
- Test conversion using GDAL from .tif -> .nc
- Test the creation of tif directory
"""


class TestBokuNDVIExporter:

    africa_url = (
        "ftp://ftp.chg.ucsb.edu/pub/org/chg/products/CHIRPS-2.0/africa_pentad/tifs/"
    )
    global_url = (
        "ftp://ftp.chg.ucsb.edu/pub/org/chg/products/CHIRPS-2.0/global_pentad/netcdf/"
    )

    @staticmethod
    def build_dummy_ftp_(tmp_path):
        conda_prefix = Path(os.environ.get("CONDA_PREFIX"))
        # make the associated directories and files
        (conda_prefix / "etc/conda/activate.d/").mkdir(parents=True, exist_okay=True)
        (conda_prefix / "etc/conda/deactivate.d/").mkdir(parents=True, exist_okay=True)
        (conda_prefix / "etc/conda/activate.d/env_vars.sh").touch()
        (conda_prefix / "etc/conda/deactivate.d/env_vars.sh").touch()

        write_to_file = """
        #!/bin/sh

        export FTP_1000='ftp://1000'
        export FTP_250='ftp://250'
        """
        with open(conda_prefix / "etc/conda/activate.d/env_vars.sh", "wb") as fp:
            fp.write(write_to_file)

        write_to_file = """
        #!/bin/sh

        unset FTP_1000
        unset FTP_250
        """
        with open(conda_prefix / "etc/conda/deactivate.d/env_vars.sh", "wb") as fp:
            fp.write(write_to_file)

    @pytest.mark.xfail(reason="GDAL not part of the testing environment")
    def test_init(self, tmp_path):
        BokuNDVIExporter(tmp_path)
        assert (
            tmp_path / "raw/boku_ndvi"
        ).exists(), "Expected a raw/boku_ndvi folder to be created!"

    @patch("os.system", autospec=True)
    def test_checkpointing(self, mock_system, tmp_path, capsys):
        # checks we don't redownload files

        exporter = BokuNDVIExporter(tmp_path)
        exporter.region_folder = exporter.output_folder / "global"
        exporter.region_folder.mkdir()

        # setup the already downloaded file
        test_filename = "testy_test.nc"
        (tmp_path / f"raw/boku_ndvi/global/{test_filename}").touch()

        exporter.wget_file(test_filename)
        captured = capsys.readouterr()

        expected_stdout = f"{test_filename} already exists! Skipping\n"
        assert (
            captured.out == expected_stdout
        ), f"Expected stdout to be {expected_stdout}, got {captured.out}"
        mock_system.assert_not_called(), "os.system was called! Should have been skipped"

    @patch("urllib.request.Request", autospec=True)
    def test_get_filenames(self, request_patch, monkeypatch, tmp_path):
        # First 1000 characters of the urllib response from the ftp for africa data,
        # pulled on May 23 2019
        request_patch.return_value = MagicMock()
        response = (
            "-rw-r--r--    1 31094    31111     4172329 Feb 01  2015 "
            "chirps-v2.0.1981.01.1.tif.gz\r\n-rw-r--r--    1 31094    "
            "31111     4133059 Feb 01  2015 chirps-v2.0.1981.01.2.tif."
            "gz\r\n-rw-r--r--    1 31094    31111     3948001 Feb 01  20"
            "15 chirps-v2.0.1981.01.3.tif.gz\r\n-rw-r--r--    1 31094    "
            "31111     3942431 Feb 01  2015 chirps-v2.0.1981.01.4.tif.g"
            "z\r\n-rw-r--r--    1 31094    31111     4176578 Feb 01  201"
            "5 chirps-v2.0.1981.01.5.tif.gz\r\n-rw-r--r--    1 31094    "
            "31111     4333592 Feb 01  2015 chirps-v2.0.1981.01.6.tif."
            "gz\r\n-rw-r--r--    1 31094    31111     4129989 Feb 02  2015"
            " chirps-v2.0.1981.02.1.tif.gz\r\n-rw-r--r--    1 31094    3"
            "1111     4067482 Feb 02  2015 chirps-v2.0.1981.02.2.tif."
            "gz\r\n-rw-r--r--    1 31094    31111     3927390 Feb 02  2015"
            " chirps-v2.0.1981.02.3.tif.gz\r\n-rw-r--r--    1 31094    311"
            "11     4208427 Feb 02  2015 chirps-v2.0.1981.02.4.tif.gz\r\n-"
            "rw-r--r--    1 31094    31111     4301096 Feb 02  2015 chirps-v"
            "2.0.1981.02.5.tif.gz\r\n"
        )

        expected_filenames = [
            "chirps-v2.0.1981.01.1.tif.gz",
            "chirps-v2.0.1981.01.2.tif.gz",
            "chirps-v2.0.1981.01.3.tif.gz",
            "chirps-v2.0.1981.01.4.tif.gz",
            "chirps-v2.0.1981.01.5.tif.gz",
            "chirps-v2.0.1981.01.6.tif.gz",
            "chirps-v2.0.1981.02.1.tif.gz",
            "chirps-v2.0.1981.02.2.tif.gz",
            "chirps-v2.0.1981.02.3.tif.gz",
            "chirps-v2.0.1981.02.4.tif.gz",
            "chirps-v2.0.1981.02.5.tif.gz",
        ]

        def mockreturn(request):
            class OpenURL:
                def read(self):
                    return response

            open_url = OpenURL()
            return open_url

        monkeypatch.setattr(urllib.request, "urlopen", mockreturn)

        exporter = BokuNDVIExporter(tmp_path)
        filenames = exporter.get_chirps_filenames(region="africa", period="pentad")

        assert filenames == expected_filenames

        # Can't get assert_called_with to work, this is a hackey workaround
        assert (
            request_patch.call_args[0][0] == self.africa_url
        ), f"Expected the urllib request to use following url: {self.africa_url}"
