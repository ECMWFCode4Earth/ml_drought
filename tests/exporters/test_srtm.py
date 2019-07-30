import unittest

from src.exporters import SRTMExporter


class TestSRTMExporter:

    def test_init(self, tmp_path):

        _ = SRTMExporter(tmp_path)
        assert (tmp_path / 'raw/srtm').exists(), 'SRTM folder not created!'

    @unittest.mock.patch('elevation.clip')
    @unittest.mock.patch('gdal.Open')
    def test_checkpointing(self, mock_elevation, mock_gdal, tmp_path):

        region_name = 'kenya'

        exporter = SRTMExporter(tmp_path)

        (tmp_path / f'raw/srtm/{region_name}.tif').touch()
        (tmp_path / f'raw/srtm/{region_name}.nc').touch()

        exporter.export(region_name=region_name)

        assert not mock_elevation.called, 'elevation.clip should not have been called!'
        assert not mock_gdal.called, 'gdal.Open should not have been called!'
