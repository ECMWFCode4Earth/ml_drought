from pathlib import Path
from typing import List, Optional
from bs4 import BeautifulSoup
import urllib.request
import numpy as np
import os
import multiprocessing

import re

from .base import BaseExporter


class NDVIExporter(BaseExporter):
    """Exports Normalised Difference Vegetation Index from NOAA

    https://www.ncei.noaa.gov/data/avhrr-land-normalized-difference-vegetation-index/access/
    """

    def __init__(self, data_folder: Path = Path('data')) -> None:
        super().__init__(data_folder)

        self.ndvi_folder = self.raw_folder / "ndvi"
        if not self.ndvi_folder.exists():
            self.ndvi_folder.mkdir()

        self.base_url = 'https://www.ncei.noaa.gov/data/' \
        'avhrr-land-normalized-difference-vegetation-index/' \
        'access'.replace(' ', '')

    @staticmethod
    def beautiful_soup_url(url: str) -> BeautifulSoup:
        # use urllib.request to read the page source
        req = urllib.request.Request(url)
        response = urllib.request.urlopen(req)
        the_page = response.read()

        # use BeautifulSoup to parse the html source
        soup = BeautifulSoup(the_page, features="lxml")

        return soup

    def get_ndvi_url_paths(self,
                           selected_years: Optional[List[int]] = None,
                           ) -> List[str]:
        # use BeautifulSoup to parse the html source
        soup = self.beautiful_soup_url(self.base_url)
        # find all links (to the years)
        years = [
            yrs.string.replace('/', '')
            for yrs in soup.find_all('a')
            if re.match(r'[0-9]{4}', yrs.string)
        ]

        # filter for selected_years
        if selected_years is not None:
            years = [y for y in years if int(y) in selected_years]

        # build the year urls
        year_urls = [
            f'{self.base_url}/{y}'
            for y in years
        ]

        # get the urls for the .nc files
        all_urls = []
        for url in year_urls:
            links = self.beautiful_soup_url(url).find_all('a')
            nc_links = [
                f'{url}/{l.string}'
                for l in links
                if '.nc' in l.string
            ]
            all_urls.extend(nc_links)

        return all_urls

    def wget_file(self, url) -> None:
        # create year subdirectories
        year = url.split('/')[-2]
        out_folder = self.ndvi_folder / year
        if not out_folder.exists():
            out_folder.mkdir(parents=True, exist_ok=True)

        # check if file already exists
        fname = url.split('/')[-1]
        if (out_folder / fname).exists():
            print(f'{fname} for {year} already donwloaded!')
            return

        os.system(f'wget -np -nH {url} -P {out_folder.as_posix()}')
        print(f'{fname} for {year} downloaded!')

    def export(self, years: Optional[List[int]] = None,
               parallel_processes: int = 1) -> None:
        """Export functionality for the NDVI product from AVHRR (NOAA)
            1981 - 2019.
        Arguments
        ----------
        years: Optional list of ints, default = None
            The years of data to download. If None, all data will be downloaded
        parallel_processes: int, default = 1
            number of processes to parallelize the downloading of data
        """
        if years is not None:
            valid_years = np.arange(1981, 2020)
            assert np.isin(years, valid_years).all(), \
                'Expected `years` argument to be in range 1981-2019'

        urls = self.get_ndvi_url_paths(selected_years=years)

        if parallel_processes <= 1:  # sequential
            for url in urls:
                self.wget_file(url)
        else:  # parallel
            pool = multiprocessing.Pool(processes=parallel_processes)
            pool.map(self.wget_file, urls)


#
