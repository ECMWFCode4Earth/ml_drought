import paramiko
from unittest.mock import patch, Mock

from src.exporters import GLEAMExporter


class TestGLEAMExporter:
    @staticmethod
    def mock_sftp_connection(transport):
        class SFTPCon:
            dirs = {
                "data": {
                    "v3.3a": {
                        "yearly": {"hello_1.nc": None},
                        "monthly": {"hello_2.nc": None},
                        "daily": {"2018": {"hello_3.nc": None}},
                    }
                }
            }

            def __init__(self):
                self.current_dir = self.dirs["data"]

            def chdir(self, dir):
                dir_list = dir.split("/")

                cur_location = self.dirs
                for subdir in dir_list:
                    if subdir != "":
                        cur_location = cur_location[subdir]

                self.current_dir = cur_location

            def listdir(self):
                return [key for key, val in self.current_dir.items()]

        return SFTPCon()

    @patch("paramiko.Transport", autospec=True)
    def test_init(self, mock_paramiko, tmp_path, monkeypatch):

        mock_paramiko.return_value = Mock()

        monkeypatch.setattr(
            paramiko.SFTPClient, "from_transport", self.mock_sftp_connection
        )

        GLEAMExporter(data_folder=tmp_path)
        assert (tmp_path / "raw/gleam").exists(), "Gleam folder not made!"

        mock_paramiko.assert_called(), "paramiko.Transport never called!"

    @patch("paramiko.Transport", autospec=True)
    def test_granularities(self, mock_paramiko, tmp_path, monkeypatch):
        mock_paramiko.return_value = Mock()
        monkeypatch.setattr(
            paramiko.SFTPClient, "from_transport", self.mock_sftp_connection
        )

        exporter = GLEAMExporter(data_folder=tmp_path)
        granularities = exporter.get_granularities()

        assert set(granularities) == {"daily", "monthly", "yearly"}

    @patch("paramiko.Transport", autospec=True)
    def test_datasets(self, mock_paramiko, tmp_path, monkeypatch):
        mock_paramiko.return_value = Mock()
        monkeypatch.setattr(
            paramiko.SFTPClient, "from_transport", self.mock_sftp_connection
        )

        exporter = GLEAMExporter(data_folder=tmp_path)

        expected = {
            "yearly": "hello_1.nc",
            "monthly": "hello_2.nc",
            "daily": "hello_3.nc",
        }

        for key, val in expected.items():

            datasets = exporter.get_datasets(granularity=key)
            assert len(datasets) == 1, "Only expected one output filename"
            output_file = datasets[0].split("/")[-1]
            assert output_file == val, f"Expected {val}, got {output_file}"

    def test_variable_to_filename(self):

        real_files = [
            "Eb_1980_2018_GLEAM_v3.3a_MO.nc",
            "Ei_1980_2018_GLEAM_v3.3a_MO.nc",
            "Ep_1980_2018_GLEAM_v3.3a_MO.nc",
            "Es_1980_2018_GLEAM_v3.3a_MO.nc",
            "Et_1980_2018_GLEAM_v3.3a_MO.nc",
            "Ew_1980_2018_GLEAM_v3.3a_MO.nc",
            "E_1980_2018_GLEAM_v3.3a_MO.nc",
            "SMroot_1980_2018_GLEAM_v3.3a_MO.nc",
            "SMsurf_1980_2018_GLEAM_v3.3a_MO.nc",
            "S_1980_2018_GLEAM_v3.3a_MO.nc",
        ]

        path = "/base/interim1/interim2"
        input_filenames = [f"{path}/{filename}" for filename in real_files]

        var_path = GLEAMExporter.variable_to_filename("SMroot", input_filenames)

        assert len(var_path) == 1, "Only expected one output filename"
        expected_filename = f"{path}/SMroot_1980_2018_GLEAM_v3.3a_MO.nc"
        assert (
            var_path[0] == expected_filename
        ), f"Expected {expected_filename}, got {var_path[0]}"

    @patch("paramiko.Transport", autospec=True)
    def test_sftppath_to_localpath(self, mock_paramiko, tmp_path, monkeypatch):
        mock_paramiko.return_value = Mock()
        monkeypatch.setattr(
            paramiko.SFTPClient, "from_transport", self.mock_sftp_connection
        )

        exporter = GLEAMExporter(data_folder=tmp_path)

        sftppath = "/data/v3.3a/yearly/Eb_1980_2018_GLEAM_v3.3a_MO.nc"
        localpath, filename = exporter.sftppath_to_localpath(sftppath)

        assert localpath == tmp_path / "raw/gleam/yearly"
        assert filename == "Eb_1980_2018_GLEAM_v3.3a_MO.nc"
