from pathlib import Path
import os
import string
import pandas as pd

from .base import BaseExporter


class ESACCIExporter(BaseExporter):
    """Exports Land Cover Maps from ESA site

    ALL (55GB .nc)
    ftp://geo10.elie.ucl.ac.be/v207/ESACCI-LC-L4-LCCS-Map-300m-P1Y-1992_2015-v2.0.7b.nc.zip

    YEARLY (300MB / yr .tif)

    LEGEND ( .csv)
    http://maps.elie.ucl.ac.be/CCI/viewer/download/ESACCI-LC-Legend.csv
    """
    @staticmethod
    def remove_punctuation(text: str) -> str:
        trans = str.maketrans('', '', string.punctuation)
        return text.lower().translate(trans)

    @staticmethod
    def read_legend() -> pd.DataFrame:
        legend_url = 'http://maps.elie.ucl.ac.be/CCI/viewer/download/ESACCI-LC-Legend.csv'
        df = pd.read_csv(legend_url, delimiter=';')
        df = df.rename(columns={'NB_LAB': 'code', 'LCCOwnLabel': 'label'})

        # standardise text (remove punctuation & lowercase)
        df['label_text'] = df['label'].apply(self.remove_punctuation)
        df = df[['code', 'label', 'label_text', 'R', 'G', 'B']]

        return df

    def wget_file(self) -> None:
        url_path = 'ftp://geo10.elie.ucl.ac.be/v207/ESACCI-LC-L4'\
            '-LCCS-Map-300m-P1Y-1992_2015-v2.0.7b.nc.zip'.replace(' ', '')

        if (self.landcover_folder / url_path.split('/')[-1]).exists():
            print(f'{filepath} already exists! Skipping')
        os.system(f'wget {url_path} -P {self.landcover_folder.as_posix()}')

    def unzip(self) -> None:
        out_name = 'ESACCI-LC-L4-LCCS-Map-300m-P1Y-1992_2015-v2.0.7b.nc'
        fname = self.landcover_folder / (out_name + '.zip')
        assert fname.exists()
        print(f'Unzipping {fname.name}')

        os.system(f'unzip {fname.as_posix()}')
        assert out_name.exists()
        print(f'{fname.name} unzipped!')

    def export(self) -> None:
        """Export functionality for the ESA CCI LandCover product
        """
        # write the download to landcover
        self.landcover_folder = self.raw_folder / 'esa_cci_landcover'
        if not self.landcover_folder.exists():
            self.landcover_folder.mkdir()

        # check if the file already exists
        fname = 'ESACCI-LC-L4-LCCS-Map-300m-P1Y-1992_2015-v2.0.7b.nc'
        if (self.landcover_folder / (fname + '.zip')).exists():
            if (self.landcover_folder / fname).exists():
                print('zip folder already exists. Unzipping')
                self.unzip()

            else:
                print('Data already downloaded!')

        # download the file using wget
        self.wget_file()

        # unzip the downloaded .zip file -> .nc
        self.unzip()

        # download the legend
        df = self.read_legend()
        df.to_csv(self.landcover_folder / 'legend.csv')
