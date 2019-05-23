from src.exporters import CHIRPSExporter


class TestCHIRPSExporter:

    def test_(self, tmp_path):
        CHIRPSExporter(tmp_path)
        assert (tmp_path / 'raw/chirps').exists(), 'Expected a raw/chirps folder to be created!'
