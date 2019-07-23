# from pathlib import Path
from unittest.mock import patch, MagicMock
import urllib.request
# import pytest
import numpy as np

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

    @patch('urllib.request.Request', autospec=True)
    def test_get_filenames(self, request_patch, monkeypatch, tmp_path):
        # First 1000 characters of the urllib response from the https,
        # pulled on July 23 2019
        request_patch.return_value = MagicMock()

        # EXPECTED response for first page (all years)
        expected_response = '<!DOCTYPE HTML PUBLIC "-//W3C//DTD ' \
            'HTML 3.2 Final//EN">\n<html>\n <head>\n<title>Index of' \
            '/data/avhrr-land-normalized-difference-vegetation-index/access</title>\n' \
            '</head>\n <body>\n<h1>Index of' \
            '/data/avhrr-land-normalized-difference-vegetation-index/access</h1>' \
            '\n<table><tr><th>&nbsp;</th><th><a' \
            'href="?C=N;O=D">Name</a></th><th><a href="?C=M;O=A">Last modified</a></th><th><a' \
            'href="?C=S;O=A">Size</a></th><th><a' \
            'href="?C=D;O=A">Description</a></th></tr><tr><th' \
            'colspan="5"><hr></th></tr>\n<tr><td valign="top">&nbsp;</td><td><a' \
            'href="/data/avhrr-land-normalized-difference-vegetation-index/">Parent' \
            'Directory</a></td><td>&nbsp;</td><td align="right">  -' \
            '</td><td>&nbsp;</td></tr>\n<tr><td valign="top">&nbsp;</td><td><a' \
            'href="1981/">1981/</a></td><td align="right">14-Jul-2019 16:09'

        # EXPECTED response for second page (1981 all .nc files)
        expected_response = '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">\n<html>\n' \
            '<head>\n  <title>Index of' \
            '/data/avhrr-land-normalized-difference-vegetation-index/access/' \
            '1981</title>\n</head>\n <body>\n<h1>Index of' \
            '/data/avhrr-land-normalized-difference-vegetation-index/access/1981' \
            '</h1>\n<table><tr><th>&nbsp;</th><th><a' \
            'href="?C=N;O=D">Name</a></th><th><a href="?C=M;O=A">Last' \
            'modified</a></th><th><a href="?C=S;O=A">Size</a></th><th><a' \
            'href="?C=D;O=A">Description</a></th></tr><tr><th' \
            'colspan="5"><hr></th></tr>\n<tr><td valign="top">&nbsp;</td><td><a' \
            'href="/data/avhrr-land-normalized-difference-vegetation-index/access/">Parent' \
            'Directory</a></td><td>&nbsp;</td><td align="right">  -' \
            '</td><td>&nbsp;</td></tr>\n<tr><td valign="top">&nbsp;</td><td><a' \
            'href="AVHRR-Land_v005_AVH13C1_NOAA-07_19810624_c20170610041337.nc">' \
            'AVHRR-Land_v005_AVH13C1_NOAA-07_19810624_c20170610041337.nc</a></td><td' \
            'align="right">12-Jul-2019 10:37  </td><td align="right">' \
            '51M</td><td>&nbsp;</td></tr>\n<tr><td valign="top">&nbsp;</td><td><a' \
            'href="AVHRR-Land_v005_AVH13C1_NOAA-07_19810625_c20170610042839.nc">' \
            'AVHRR-Land_v005_AVH13C1_NOAA-07_19810625_c20170610042839.nc</a></td><td' \
            'align="right">12-Jul-2019 10:37  </td><td align="right">' \
            '59M</td><td>&nbsp;</td></tr>\n<tr><td valign="top">&nbsp;</td><td><a'

        expected_urls = [
            'https://www.ncei.noaa.gov/data/avhrr-land-normalized-'
            'difference-vegetation-index/access/1981/AVHRR-Land_'
            'v005_AVH13C1_NOAA-07_19810624_c20170610041337.nc',
            'https://www.ncei.noaa.gov/data/avhrr-land-normalized-'
            'difference-vegetation-index/access/1981/AVHRR-Land_'
            'v005_AVH13C1_NOAA-07_19810625_c20170610042839.nc',
        ]

        # HOW TO MOCK beautiful_soup_url function
        def mockreturn(request):
            class OpenURL:
                def read(self):
                    return expected_response
            open_url = OpenURL()
            return open_url

        # i want to patch this ::L34-L36 $ the_page = response.read()
        monkeypatch.setattr(urllib.request, 'urlopen', mockreturn)

        exporter = NDVIExporter(tmp_path)
        filenames = exporter.get_ndvi_url_paths(selected_years=np.arange(1981, 1985))

        assert filenames is not None
        assert expected_urls is not None
