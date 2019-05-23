from bs4 import BeautifulSoup
import urllib.request
import os
import warnings
import multiprocessing
from pathlib import Path
from .base import BaseExporter

from typing import List, Optional


class CHIRPSExporter(BaseExporter):
    """Exports precip from the Climate Hazards group site

    # 0.5degree
    ftp://ftp.chg.ucsb.edu/pub/org/chg/products/CHIRPS-2.0/global_pentad/netcdf/
    # 0.25degree
    ftp://ftp.chg.ucsb.edu/pub/org/chg/products/CHIRPS-2.0/africa_pentad/tifs/
    """

    def __init__(self, data_folder: Path = Path('data')) -> None:
        super().__init__(data_folder)

        self.chirps_folder = self.raw_folder / "chirps"
        if not self.chirps_folder.exists():
            self.chirps_folder.mkdir()

        self.base_url = 'ftp://ftp.chg.ucsb.edu/pub/org/chg'

    def get_url(self, region: str = 'africa') -> str:
        url = f'/products/CHIRPS-2.0/{region}_pentad/{"tifs" if region == "africa" else "netcdf"}/'

        return self.base_url + url

    def get_chirps_filenames(self, years: Optional[List[int]] = None,
                             region: str = 'africa') -> List[str]:
        """
        ftp://ftp.chg.ucsb.edu/pub/org/chg/products/
            CHIRPS-2.0/global_pentad/netcdf/

        https://github.com/datamission/WFP/blob/master/Datasets/CHIRPS/get_chirps.py
        """
        url = self.get_url(region)

        # use urllib.request to read the page source
        req = urllib.request.Request(url)
        response = urllib.request.urlopen(req)
        the_page = response.read()

        # use BeautifulSoup to parse the html source
        page = str(BeautifulSoup(the_page, features="lxml"))

        # split the page to get the filenames as a list
        firstsplit = page.split('\r\n')  # split the newlines
        secondsplit = [x.split(' ') for x in firstsplit]  # split the spaces
        flatlist = [item for sublist in secondsplit for item in sublist]  # flatten
        chirpsfiles = [x for x in flatlist if 'chirps' in x]

        # extract only the years of interest
        if years is not None:
            chirpsfiles = [
                f for f in chirpsfiles if any(
                    [f'.{yr}.' in f for yr in years]
                )
            ]
        return chirpsfiles

    def wget_file(self, filepath: str) -> None:
        if (self.chirps_folder / filepath).exists():
            print(f'{filepath} already exists! Skipping')
        else:
            os.system(f"wget -np -nH --cut-dirs 7 {filepath} \
                -P {self.chirps_folder.as_posix()}")

    def download_chirps_files(self,
                              chirps_files: List[str],
                              region: str = 'africa',
                              parallel: bool = False) -> None:
        """ download the chirps files using wget """
        # build the base url

        url = self.get_url(region)

        filepaths = [url + f for f in chirps_files]

        if parallel:
            pool = multiprocessing.Pool(processes=100)
            pool.map(self.wget_file, filepaths)
        else:
            for file in filepaths:
                self.wget_file(file)

    def export(self, years: Optional[List[int]] = None,
               region: str = 'africa',
               parallel: bool = False) -> None:
        """Export functionality for the CHIRPS precipitation product

        Arguments
        ----------
        years: Optional list of ints, default = None
            The years of data to download. If None, all data will be downloaded
        region: str {'africa', 'global'}, default = 'africa'
            The dataset region to download. If global, a netcdf file is downloaded.
            If africa, a tif file is downloaded
        parallel: bool, default = False
            Whether to parallelize the downloading of data
        """

        if years is not None:
            assert min(years) >= 1981, f"Minimum year cannot be less than 1981. " \
                f"Currently: {min(years)}"
            if max(years) > 2020:
                warnings.warn(f"Non-breaking change: max(years) is: {max(years)}. "
                              f"But no files later than 2019")

        # get the filenames to be downloaded
        chirps_files = self.get_chirps_filenames(years, region)

        # check if they already exist
        existing_files = [
            f.as_posix().split('/')[-1]
            for f in self.chirps_folder.glob('*.nc')
        ]
        chirps_files = [f for f in chirps_files if f not in existing_files]

        # download files in parallel
        self.download_chirps_files(chirps_files, region, parallel)
