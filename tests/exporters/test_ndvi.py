from unittest.mock import patch

from src.exporters import NDVIExporter


class TestNDVIExporter:
    def test_init(self, tmp_path):
        e = NDVIExporter(tmp_path)

        assert e.ndvi_folder.name == 'ndvi'
        assert (tmp_path / 'raw' / 'ndvi').exists()

    @patch('os.system', autospec=True)
    def test_checkpointing(self, mock_system, tmp_path, capsys):
        # checks we don't redownload files
        exporter = NDVIExporter(tmp_path)

        # setup the already downloaded file
        test_filename = '1981/testy_test.nc'
        (tmp_path / 'raw/ndvi/1981').mkdir(parents=True, exist_ok=True)
        (tmp_path / f'raw/ndvi/{test_filename}').touch()

        exporter.wget_file(test_filename)
        captured = capsys.readouterr()

        expected_stdout = f'testy_test.nc for 1981 already donwloaded!\n'
        assert captured.out == expected_stdout, \
            f'Expected stdout to be {expected_stdout}, got {captured.out}'
        mock_system.assert_not_called(), 'os.system was called! Should have been skipped'

    @patch('os.system')
    def test_beautiful_soup_regex_parse(self, mock_system, tmp_path):
        exporter = NDVIExporter(tmp_path)
        files = exporter.get_ndvi_url_paths(selected_years=[1981])

        # check that all netcdf files
        assert all([f[-3:] == '.nc' for f in files])

        # check base of string
        base_url_str = 'https://www.ncei.noaa.gov/data/' \
            'avhrr-land-normalized-difference-vegetation-index/access/1981/'
        assert all([f.split('AVHRR')[0] == base_url_str for f in files])

        # check got 31 December
        timestamp = '19811231'
        assert files[-1].split('_')[-2] == timestamp
