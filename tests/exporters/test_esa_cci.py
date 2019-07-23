from pathlib import Path
from unittest.mock import patch
from src.exporters import ESACCIExporter

class TestEsaCciExporter:
    def test_init(self):
        data_path = Path(tmp_path / 'data')
        e = ESACCIExporter(data_path)

        assert e.raw_folder.name == 'esa_cci_landcover'
        assert (data_path / 'raw' / 'esa_cci_landcover').exists()

    @patch('os.system', autospec=True)
    def test_checkpointing(self, mock_system, tmp_path, capsys):
        # checks we don't redownload files
        exporter = ESACCIExporter(tmp_path)

        # setup the already downloaded file
        test_filename = 'testy_test.nc'
        (tmp_path / f'raw/esa_cci_landcover/{test_filename}').touch()

        exporter.wget_file(test_filename)
        captured = capsys.readouterr()

        expected_stdout = f'{test_filename} already exists! Skipping\n'
        assert captured.out == expected_stdout, \
            f'Expected stdout to be {expected_stdout}, got {captured.out}'
        mock_system.assert_not_called(), 'os.system was called! Should have been skipped'

    @unittest.mock.patch('os.system')
    def test_wget_file(self, mock_system, tmp_path, capsys):
        # tests the write call made to os.system
        exporter = ESACCIExporter(tmp_path)

        url_path = 'ftp://geo10.elie.ucl.ac.be/v207/ESACCI-LC-L4-LCCS-Map-'\
            '300m-P1Y-1992_2015-v2.0.7b.nc.zip'
        folder = (tmp_path / 'raw' / 'esa_cci_landcover').as_posix()
        mock_system.assert_called_once_with(
            f'wget {url_path} -P {folder}'
        )

    def test_read_legend(self, mock_system, tmp_path, capsys):
        legend_url = b'NB_LAB;LCCOwnLabel;R;G;B\r\n0;No data;0;0;0'\
        '\r\n10;Cropland, rainfed;255;255;100\r\n11;Herbaceous cover'\
        ';255;255;100\r\n12;Tree or shrub cover;255;255;0\r\n20;Cropland,'\
        ' irrigated or post-flooding;170;240;240\r\n30;Mosaic cropland '\
        '(>50%) / natural vegetation (tree, shrub, herbaceous cover) '\
        '(<50%);220;240;100\r\n40;Mosaic natural vegetation (tree, '\
        'shrub, herbaceous cover) (>50%) / cropland (<50%) ;200;200;'\
        '100\r\n50;Tree cover, broadleaved, evergreen, closed to open '\
        '(>15%);0;100;0\r\n60;Tree cover, broadleaved, deciduous, '\
        'closed to open (>15%);0;160;0\r\n61;Tree cover, broadleaved,'\
        ' deciduous, closed (>40%);0;160;0\r\n62;Tree cover, broadleaved,'\
        ' deciduous, open (15-40%);170;200;0\r\n70;Tree cover, needleleaved,'\
        ' evergreen, closed to open (>15%);0;60;0\r\n71;Tree cover, '\
        'needleleaved, evergreen, closed (>40%);0;60;0\r\n72;Tree cover,'\
        ' needleleaved, evergreen, open (15-40%);0;80;0\r\n80;Tree cover,'\
        ' needleleaved, deciduous, closed to open (>15%);40;80;0\r\n81;'\
        'Tree cover, needleleaved, deciduous, closed (>40%);40;80;0\r\n82;'\
        'Tree cover, needleleaved, deciduous, open (15-40%);40;100;0\r\n90;'\
        'Tree cover, mixed leaf type (broadleaved and needleleaved);120;130;0'\
        '\r\n100;Mosaic tree and shrub (>50%) / herbaceous cover (<50%)'\
        ';140;160;0\r\n110;Mosaic herbaceous cover (>50%) / tree and shrub'\
        ' (<50%);190;150;0\r\n120;Shrubland;150;100;0\r\n121;Shrubland '\
        'evergreen;120;75;0\r\n122;Shrubland deciduous;150;'\
        '100;0\r\n130;Grassland;255;180;50\r\n140;Lichens and '\
        'mosses;255;220;210\r\n150;Sparse vegetation (tree, shrub, '\
        'herbaceous cover) (<15%);255;235;175\r\n151;Sparse tree '\
        '(<15%);255;200;100\r\n152;Sparse shrub (<15%);255;210;120'\
        '\r\n153;Sparse herbaceous cover (<15%);255;235;175\r\n160;'\
        'Tree cover, flooded, fresh or brakish water;0;120;90\r\n170;'\
        'Tree cover, flooded, saline water;0;150;120\r\n180;Shrub or '\
        'herbaceous cover, flooded, fresh/saline/brakish water;0;220;'\
        '130\r\n190;Urban areas;195;20;0\r\n200;Bare areas;255;245;215'\
        '\r\n201;Consolidated bare areas;220;220;220\r\n202;Unconsolidated '\
        'bare areas;255;245;215\r\n210;Water bodies;0;70;200\r\n220;'\
        'Permanent snow and ice;255;255;255\r\n'
