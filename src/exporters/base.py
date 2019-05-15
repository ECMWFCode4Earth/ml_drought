from pathlib import Path
from collections import namedtuple


Region = namedtuple('Region', ['name', 'lonmin', 'lonmax', 'latmin', 'latmax'])


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

    @staticmethod
    def get_kenya() -> Region:
        """This pipeline is focused on drought prediction in Kenya.
        This function allows Kenya's bounding box to be easily accessed
        by all exporters.
        """
        return Region(name='kenya', lonmin=33.501, lonmax=42.283,
                      latmin=-5.202, latmax=6.002)
