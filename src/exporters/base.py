from pathlib import Path


class BaseExporter:
    """Base for all exporter classes

    Attributes:
    ----------
    data_folder: Path, default: Path('data')
        The location of the data folder.
    """

    def __init__(self, data_folder: Path = Path('data')) -> None:

        self.data_folder = data_folder
        self.raw_folder = self.data_folder / 'raw'
        if not self.raw_folder.exists():
            self.raw_folder.mkdir()
