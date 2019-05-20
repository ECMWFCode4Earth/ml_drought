from typing import List, Optional
from bs4 import BeautifulSoup
import urllib.request
import os

class CHIRPSExporter(BaseExporter):
    """Exports precip from the Climate Hazards group site

    ftp://ftp.chg.ucsb.edu/pub/org/chg/products/CHIRPS-2.0/global_pentad/netcdf/
    """

    self.chirps_folder = self.raw_folder / "chirps"
    if not self.chirps_folder.exists():
        self.chirps_folder.mkdir()

    @staticmethod
    def get_chirps_filenames(years: List[int]) -> List:
        """
        ftp://ftp.chg.ucsb.edu/pub/org/chg/products/
            CHIRPS-2.0/global_pentad/netcdf/

        https://github.com/datamission/WFP/blob/master/Datasets/CHIRPS/get_chirps.py
        """
        base_url = 'ftp://ftp.chg.ucsb.edu'
        url = base_url + '/pub/org/chg/products/CHIRPS-2.0/global_pentad/netcdf/'

        # use urllib.request to read the page source
        req = urllib.request.Request(url)
        response = urllib.request.urlopen(req)
        the_page = response.read()

        # use BeautifulSoup to parse the html source
        page = str(BeautifulSoup(the_page))

        # split the page to get the filenames as a list
        firstsplit=page.split('\r\n')  # split the newlines
        secondsplit = [x.split(' ') for x in firstsplit]  # split the spaces
        flatlist = [item for sublist in secondsplit for item in sublist]  # flatten
        chirpsfiles = [x for x in flatlist if 'chirps' in x]

        # select the years of interest
        chirpsfiles = [x for x in chirpsfiles if str()]
        # extract only the years of interest
        years = [str(yr) for yr in years]
        chirpsfiles = [
            f for f in chirpsfiles if any(
                [f".{yr}." in f for yr in years]
            )
        ]
        return chirpsfiles

    @staticmethod
    def wget_file(filepath: str) -> None:
        os.system(f"wget -np -nH --cut-dirs 7 {filepath} -P {self.chirps_folder.as_posix()}")


    def download_chirps_files(chirps_files: List[str]) -> None:
        """ download the chirps files using wget """
        # build the base url
        base_url = 'ftp://ftp.chg.ucsb.edu/pub/org/chg'
        base_url += '/products/CHIRPS-2.0/global_pentad/netcdf/'
        if base_url[-1] != '/':
            base_url += '/'

        filepaths = [base_url + f for f in chirps_files]

        pool = multiprocessing.pool(processes=100)
        pool.map(wget_file, filepaths)

    @staticmethod
    def get_default_years() -> List[int]:
        """ returns the default arguments for no. years """
        years = [yr for yr in range(1981, 2020)]

        return years

    def export(self, years: List[int] = None):
        if years is None:
            years = self.get_default_years()

        assert min(years) >= 1981, f"Minimum year cannot be less than 1981.\
            Currently: {min(years)}"
        if max(years) > 2020:
            warnings.warn(f"Non-breaking change: max(years) is:{ max(years)}. But no \
            files later than 2019")

        # get the filenames to be downloaded
        chirps_files = self.get_chirps_filenames(years)

        # download files in parallel
        download_chirps_files(chirps_files)

        return
