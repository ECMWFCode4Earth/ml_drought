from pathlib import Path

from typing import cast, List, Union, Tuple

from .base import BaseExporter

paramiko = None


class GLEAMExporter(BaseExporter):
    """Download data from the Global Land Evaporation Amsterdam Model
    (gleam.eu)

    Access information can be found at gleam.eu
    """

    dataset = "gleam"

    def __init__(self, data_folder: Path = Path("data")) -> None:
        super().__init__(data_folder)

        global paramiko
        if paramiko is None:
            import paramiko

        password = "v33_GLEAM2019#aw"
        username = "gleamuser"
        host = "hydras.ugent.be"
        port = 2225

        transport = paramiko.Transport((host, port))  # type: ignore
        transport.connect(username=username, password=password)
        self.sftp = paramiko.SFTPClient.from_transport(transport)  # type: ignore
        self.base_sftp_path: str = "/data/v3.3a/"

    def get_granularities(self) -> List[str]:
        self.sftp.chdir(self.base_sftp_path)
        return self.sftp.listdir()

    def get_datasets(self, granularity: str = "monthly") -> List[str]:
        granularity_path = f"{self.base_sftp_path}/{granularity}"
        self.sftp.chdir(granularity_path)

        granularity_listdir = self.sftp.listdir()

        datasets: List[str] = []
        if granularity == "daily":
            for year in granularity_listdir:
                year_path = f"{granularity_path}/{year}"
                self.sftp.chdir(year_path)
                subfiles = self.sftp.listdir()
                for subfile in subfiles:
                    datasets.append(f"{year_path}/{subfile}")
        else:
            datasets.extend(
                [f"{granularity_path}/{subfile}" for subfile in granularity_listdir]
            )

        return datasets

    @staticmethod
    def variable_to_filename(variable: str, datasets: List[str]) -> List[str]:
        output_datasets: List[str] = []

        for filepath in datasets:
            filename = filepath.split("/")[-1]
            variable_name = filename.split("_")[0]
            if variable_name == variable:
                output_datasets.append(filepath)

        return output_datasets

    def sftppath_to_localpath(self, sftppath: str) -> Tuple[Path, str]:

        stem = sftppath[len(self.base_sftp_path) :]
        filename = sftppath.split("/")[-1]
        return self.output_folder / stem[: -len(filename)].strip("/"), filename

    def export(self, variables: Union[str, List[str]], granularity: str) -> None:

        acceptable_granularities = set(self.get_granularities())
        assert granularity in acceptable_granularities, (
            f"{granularity} not an acceptable granularity! "
            f"Must be one of {acceptable_granularities}"
        )

        if type(variables) == str:
            variables = cast(List[str], [variables])

        for variable in variables:
            relevant_datasets = self.variable_to_filename(
                variable, self.get_datasets(granularity)
            )
            if len(relevant_datasets) == 0:
                print("No files found! Check your variable names")

            for dataset in relevant_datasets:
                localpath, filename = self.sftppath_to_localpath(dataset)
                print(f"Downloading {dataset} to {localpath}")
                localpath.mkdir(parents=True, exist_ok=True)
                self.sftp.get(dataset, str(localpath / filename))
