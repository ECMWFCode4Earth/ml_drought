import os
import string
import pandas as pd
from pathlib import Path
import zipfile

from .base import BaseExporter


class ESACCIExporter(BaseExporter):
    """Exports Land Cover Maps from ESA site

    ALL (55GB .nc)
    ftp://geo10.elie.ucl.ac.be/v207/ESACCI-LC-L4-LCCS-Map-300m-P1Y-1992_2015-v2.0.7b.nc.zip

    YEARLY (300MB / yr .tif)

    LEGEND ( .csv)
    http://maps.elie.ucl.ac.be/CCI/viewer/download/ESACCI-LC-Legend.csv
    """

    target_url: str = "ftp://geo10.elie.ucl.ac.be/v207"
    target_file: str = "ESACCI-LC-L4-LCCS-Map-300m-P1Y-1992_2015-v2.0.7b.nc"
    legend_url: str = "http://maps.elie.ucl.ac.be/CCI/viewer/download/ESACCI-LC-Legend.csv"

    def __init__(self, data_folder: Path = Path("data")) -> None:
        super().__init__(data_folder)
        # write the download to landcover
        self.landcover_folder = self.raw_folder / "esa_cci_landcover"
        if not self.landcover_folder.exists():
            self.landcover_folder.mkdir()

    @staticmethod
    def remove_punctuation(text: str) -> str:
        rm_punctuation = str.maketrans("", "", string.punctuation)
        rm_digits = str.maketrans("", "", string.digits)
        return (
            text.lower()
            .translate(rm_punctuation)
            .translate(rm_digits)
            .rstrip()
            .replace("   ", " ")
        )

    def read_legend(self) -> pd.DataFrame:
        df = pd.read_csv(self.legend_url, delimiter=";")
        df = df.rename(columns={"NB_LAB": "code", "LCCOwnLabel": "label"})

        # standardise text (remove punctuation & lowercase)
        df["label_text"] = df["label"].apply(self.remove_punctuation)
        df = df[["code", "label", "label_text", "R", "G", "B"]]

        return df

    def wget_file(self) -> None:
        url_path = f"{self.target_url}/{self.target_file}".replace(" ", "")

        output_file = self.landcover_folder / url_path.split("/")[-1]
        if output_file.exists():
            print(f"{output_file} already exists! Skipping")
            return None
        os.system(f"wget {url_path}.zip -P {self.landcover_folder.as_posix()}")

    def unzip(self, python_only: bool = True) -> None:
        """https://stackoverflow.com/a/3451150/9940782
        """
        fname = self.landcover_folder / (self.target_file + ".zip")
        assert fname.exists()
        print(f"Unzipping {fname.name}")
        if python_only:
            with zipfile.ZipFile(fname, "r") as zip_ref:
                zip_ref.extractall(fname.parents[0])
        else:  # ECMWF machine doesn't have the bash `unzip` utility
            os.system(
                f"unzip {fname.as_posix()} -d {self.landcover_folder.resolve().as_posix()}"
            )
        print(f"{fname.name} unzipped!")

    def export(self) -> None:
        """Export functionality for the ESA CCI LandCover product
        """
        # check if the file already exists
        if (self.landcover_folder / self.target_file).exists() and (
            self.landcover_folder / "legend.csv"
        ).exists():
            print("Data already downloaded!")

        else:
            self.wget_file()
            self.unzip()
            # download the legend
            df = self.read_legend()
            df.to_csv(self.landcover_folder / "legend.csv")
