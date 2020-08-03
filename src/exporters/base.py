from pathlib import Path

from typing import Optional
from ..utils import Region, get_kenya, region_lookup

__all__ = ["BaseExporter", "Region", "get_kenya", "region_lookup"]


class BaseExporter:
    """Base for all exporter classes

    Attributes:
    ----------
    data_folder: Path, default: Path('data')
        The location of the data folder.
    """

    dataset: Optional[str] = None

    def __init__(self, data_folder: Path = Path("data")) -> None:

        self.data_folder = data_folder
        self.raw_folder = self.data_folder / "raw"
        if not self.raw_folder.exists():
            self.raw_folder.mkdir(exist_ok=True)

        if self.dataset is not None:
            self.output_folder = self.raw_folder / self.dataset
            if not self.output_folder.exists():
                self.output_folder.mkdir()
