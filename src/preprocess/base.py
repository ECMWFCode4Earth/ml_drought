from pathlib import Path

from ..utils import Region, get_kenya


__all__ = ['BasePreProcessor', 'Region', 'get_kenya']


class BasePreProcessor:
    """Base for all pre-processor classes

    Attributes:
    ----------
    data_folder: Path, default: Path('data')
        The location of the data folder.
    """
    def __init__(self, data_folder: Path = Path('data')) -> None:
        self.data_folder = data_folder
        self.raw_folder = self.data_folder / 'raw'
        self.interim_folder = self.data_folder / 'interim'

        if not self.interim_folder.exists():
            self.interim_folder.mkdir()
