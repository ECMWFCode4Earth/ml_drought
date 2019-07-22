# from pathlib import Path
# from unittest.mock import patch
# import pytest
# import re
#
# from src.exporters import EsaCciExporter

class TestEsaCciExporter:
    def test_init(self):
        data_path = Path(tmp_path / 'data')
        e = EsaCciExporter(data_path)

        assert e.raw_folder.name == 'esa_cci_landcover'
        assert (data_path / 'raw' / 'esa_cci_landcover').exists()

    @patch('os.system', autospec=True)
    def test_checkpointing(self, mock_system, tmp_path, capsys):
        # checks we don't redownload files

        exporter = CHIRPSExporter(tmp_path)
        exporter.region_folder = exporter.chirps_folder / 'global'
        exporter.region_folder.mkdir()

        # setup the already downloaded file
        test_filename = 'testy_test.nc'
        (tmp_path / f'raw/chirps/global/{test_filename}').touch()

        exporter.wget_file(test_filename)
        captured = capsys.readouterr()

        expected_stdout = f'{test_filename} already exists! Skipping\n'
        assert captured.out == expected_stdout, \
            f'Expected stdout to be {expected_stdout}, got {captured.out}'
        mock_system.assert_not_called(), 'os.system was called! Should have been skipped'
