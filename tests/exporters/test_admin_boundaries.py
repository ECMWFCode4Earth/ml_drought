import unittest.mock
import unittest

from src.exporters import KenyaAdminExporter


class TestKenyaAdminExporter:
    def test_init(self, tmp_path):
        _ = KenyaAdminExporter(tmp_path)
        assert (
            tmp_path / "raw/boundaries/kenya"
        ).exists(), "boundaries/kenya folder not created!"

    @unittest.mock.patch("os.system")
    def test_checkpointing(self, mock_system, tmp_path):

        exporter = KenyaAdminExporter(tmp_path)

        (tmp_path / f"raw/boundaries/kenya/ken_admin1.zip").touch()
        (tmp_path / f"raw/boundaries/kenya/kendistricts.zip").touch()
        (tmp_path / f"raw/boundaries/kenya/kenya_wards.zip").touch()
        (tmp_path / f"raw/boundaries/kenya/kendivisions.zip").touch()
        (tmp_path / f"raw/boundaries/kenya/kenlocations.zip").touch()
        (tmp_path / f"raw/boundaries/kenya/kensublocations.zip").touch()

        exporter.export()

        assert not mock_system.called, "os.system should not have been called!"
