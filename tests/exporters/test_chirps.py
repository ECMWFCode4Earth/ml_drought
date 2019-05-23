from unittest.mock import patch

from src.exporters import CHIRPSExporter


class TestCHIRPSExporter:

    def test_init(self, tmp_path):
        CHIRPSExporter(tmp_path)
        assert (tmp_path / 'raw/chirps').exists(), 'Expected a raw/chirps folder to be created!'

    @patch('os.system', autospec=True)
    def test_checkpointing(self, mock_system, tmp_path, capsys):
        # checks we don't redownload files

        exporter = CHIRPSExporter(tmp_path)

        # setup the already downloaded file
        test_filename = 'testy_test.nc'
        (tmp_path / f'raw/chirps/{test_filename}').touch()

        exporter.wget_file(test_filename)
        captured = capsys.readouterr()

        expected_stdout = f'{test_filename} already exists! Skipping\n'
        assert captured.out == expected_stdout, \
            f'Expected stdout to be {expected_stdout}, got {captured.out}'
        mock_system.assert_not_called(), 'os.system was called! Should have been skipped'
