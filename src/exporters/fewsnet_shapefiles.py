from pathlib import Path
import os
from typing import List

from .base import BaseExporter


class FEWSNetExporter(BaseExporter):
    """Export FEWSNet data

    https://fews.net/

    TODO: need to use Selenium to navigate this page?
    https://fews.net/data
    """

    data_str: str

    def __init__(self, data_folder: Path = Path("data")) -> None:
        super().__init__(data_folder)
        # write the download to landcover
        self.output_dir = self.raw_folder / "boundaries" / self.data_str
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def wget_file(self, url_path: str) -> None:
        output_file = self.output_dir / url_path.split("/")[-1]
        if output_file.exists():
            print(f"{output_file} already exists! Skipping")
            return None

        os.system(f"wget {url_path} -P {self.output_dir.as_posix()}")

    def unzip(self, fname: Path) -> None:
        print(f"Unzipping {fname.name}")

        os.system(f"unzip {fname.as_posix()} -d {self.output_dir.resolve().as_posix()}")
        print(f"{fname.name} unzipped!")


class FEWSNetKenyaLivelihoodExporter(FEWSNetExporter):
    data_str = "livelihood_zones"
    url: str = "http://shapefiles.fews.net.s3.amazonaws.com/LHZ/FEWS_NET_LH_World.zip"

    def export(self) -> None:
        """Export functionality for the FEWSNET Livelihood Zones as .shp files
        """

        fname = self.url.split("/")[-1]
        # check if the file already exists
        if (self.output_dir / fname).exists():
            print("Data already downloaded!")

        else:
            self.wget_file(url_path=self.url)
            self.unzip(fname=(self.output_dir / fname))
