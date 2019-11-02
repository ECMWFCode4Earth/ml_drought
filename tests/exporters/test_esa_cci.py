import unittest
from io import StringIO
import pandas as pd

from unittest.mock import patch
from src.exporters import ESACCIExporter


class TestEsaCCIExporter:
    def test_init(self, tmp_path):
        e = ESACCIExporter(tmp_path)

        assert e.landcover_folder.name == "esa_cci_landcover"
        assert (tmp_path / "raw" / "esa_cci_landcover").exists()

    @patch("os.system", autospec=True)
    def test_wget_checkpointing(self, mock_system, tmp_path, capsys):
        # checks we don't redownload files
        exporter = ESACCIExporter(tmp_path)

        # setup the already downloaded file
        test_filename = exporter.target_file
        (tmp_path / f"raw/esa_cci_landcover/{test_filename}").touch()
        (tmp_path / f"raw/esa_cci_landcover/legend.csv").touch()

        exporter.wget_file()
        captured = capsys.readouterr()

        expected_stdout = f"{test_filename} already exists! Skipping\n"
        assert (
            expected_stdout in captured.out
        ), f"Expected stdout to be {expected_stdout}, got {captured.out}"
        mock_system.assert_not_called(), "os.system was called! Should have been skipped"

    @patch("os.system", autospec=True)
    def test_wget_export(self, mock_system, tmp_path, capsys):
        # checks we don't redownload files
        exporter = ESACCIExporter(tmp_path)

        # setup the already downloaded file
        test_filename = exporter.target_file
        (tmp_path / f"raw/esa_cci_landcover/{test_filename}").touch()
        (tmp_path / f"raw/esa_cci_landcover/legend.csv").touch()

        exporter.export()
        captured = capsys.readouterr()

        expected_stdout = "Data already downloaded!"
        assert (
            expected_stdout in captured.out
        ), f"Expected stdout to be {expected_stdout}, got {captured.out}"
        mock_system.assert_not_called(), "os.system was called! Should have been skipped"

    @unittest.mock.patch("os.system")
    def test_wget_file(self, mock_system, tmp_path):
        # tests the write call made to os.system
        exporter = ESACCIExporter(tmp_path)

        url_path = (
            "ftp://geo10.elie.ucl.ac.be/v207/ESACCI-LC-L4-LCCS-Map-"
            "300m-P1Y-1992_2015-v2.0.7b.nc.zip"
        )
        folder = (tmp_path / "raw" / "esa_cci_landcover").as_posix()

        exporter.wget_file()
        mock_system.assert_called_once_with(f"wget {url_path} -P {folder}")

    def test_read_legend(self, tmp_path, monkeypatch):
        legend_url = (
            "NB_LAB;LCCOwnLabel;R;G;B\r\n0;No data;0;0;0"
            "\r\n10;Cropland, rainfed;255;255;100\r\n11;Herbaceous cover"
            ";255;255;100\r\n12;Tree or shrub cover;255;255;0\r\n20;Cropland,"
            " irrigated or post-flooding;170;240;240"
        )

        output_csv = pd.read_csv(StringIO(legend_url), delimiter=";")

        def mockread(url, delimiter):
            return output_csv

        monkeypatch.setattr(pd, "read_csv", mockread)

        exporter = ESACCIExporter(tmp_path)
        legend = exporter.read_legend()
        assert (legend.columns == ["code", "label", "label_text", "R", "G", "B"]).all()
