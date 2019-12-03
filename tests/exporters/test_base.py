from src.exporters.base import BaseExporter


class TestBase:
    def test_make_raw(self, tmp_path):

        _ = BaseExporter(tmp_path)
        assert (tmp_path / "raw").exists(), "Expected a raw folder to be generated!"
