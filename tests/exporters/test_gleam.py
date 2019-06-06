import paramiko
from unittest.mock import patch, Mock

from src.exporters import GLEAMExporter


class TestGLEAMExporter:

    @staticmethod
    def mock_sftp_connection(transport):

        class SFTPCon:
            dirs = {
                'data': {'v3.3a': {'yearly': {'hello_1.nc': None},
                                   'monthly': {'hello_2.nc': None},
                                   'daily': {'2018': {'hello_3.nc': None}}
                                   }
                         }
            }

            def __init__(self):
                self.current_dir = self.dirs['data']

            def chdir(self, dir):
                dir_list = dir.split('/')

                cur_location = self.dirs
                for subdir in dir_list:
                    if subdir != '':
                        cur_location = cur_location[subdir]

                self.current_dir = cur_location

            def listdir(self):
                return [key for key, val in self.current_dir.items()]

        return SFTPCon()

    @patch('paramiko.Transport', autospec=True)
    def test_init(self, mock_paramiko, tmp_path, monkeypatch):

        mock_paramiko.return_value = Mock()

        monkeypatch.setattr(paramiko.SFTPClient, 'from_transport',
                            self.mock_sftp_connection)

        GLEAMExporter(username='Bob', password='123', host='453', port=789,
                      data_folder=tmp_path)
        assert (tmp_path / 'raw/gleam').exists(), 'Gleam folder not made!'

        mock_paramiko.assert_called(), 'paramiko.Transport never called!'

    @patch('paramiko.Transport', autospec=True)
    def test_granularities(self, mock_paramiko, tmp_path, monkeypatch):
        mock_paramiko.return_value = Mock()
        monkeypatch.setattr(paramiko.SFTPClient, 'from_transport',
                            self.mock_sftp_connection)

        exporter = GLEAMExporter(username='Bob', password='123', host='453',
                                 port=789, data_folder=tmp_path)
        granularities = exporter.get_granularities()

        assert set(granularities) == {'daily', 'monthly', 'yearly'}

    @patch('paramiko.Transport', autospec=True)
    def test_datasets(self, mock_paramiko, tmp_path, monkeypatch):
        mock_paramiko.return_value = Mock()
        monkeypatch.setattr(paramiko.SFTPClient, 'from_transport',
                            self.mock_sftp_connection)

        exporter = GLEAMExporter(username='Bob', password='123', host='453',
                                 port=789, data_folder=tmp_path)

        expected = {'yearly': 'hello_1.nc', 'monthly': 'hello_2.nc', 'daily': 'hello_3.nc'}

        for key, val in expected.items():

            datasets = exporter.get_datasets(granularity=key)
            assert len(datasets) == 1, 'Only expected one output filename'
            output_file = datasets[0].split('/')[-1]
            assert output_file == val, f'Expected {val}, got {output_file}'
