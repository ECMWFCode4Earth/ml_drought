import paramiko
from pathlib import Path

from typing import List

from .base import BaseExporter


class GLEAMExporter(BaseExporter):
    """Download data from the Global Land Evaporation Amsterdam Model
    (gleam.eu)

    Access information can be found at gleam.eu
    """

    def __init__(self, username: str, password: str,
                 host: str, port: int,
                 data_folder: Path = Path('data')) -> None:
        super().__init__(data_folder)

        transport = paramiko.Transport((host, port))
        transport.connect(username=username, password=password)
        self.sftp = paramiko.SFTPClient.from_transport(transport)
        self.base_sftp_path: str = '/data/v3.3a/'

    def get_granularities(self) -> List[str]:
        self.sftp.chdir(self.base_sftp_path)
        return self.sftp.listdir()

    def get_datasets(self, granularity: str = 'monthly') -> List[str]:
        granularity_path = f'{self.base_sftp_path}/{granularity}'
        self.sftp.chdir(granularity_path)

        granularity_listdir = self.sftp.listdir()

        datasets: List[str] = []
        if granularity == 'daily':
            for year in granularity_listdir:
                year_path = f'{granularity_path}/{year}'
                self.sftp.chdir(year_path)
                subfiles = self.sftp.listdir()
                for subfile in subfiles:
                    datasets.append(f'{year_path}/{subfile}')
        else:
            datasets.extend([f'{granularity_path}/{subfile}' for subfile in granularity_listdir])

        return datasets
