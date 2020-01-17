"""
We use the following method to hide API keys:
https://towardsdatascience.com/how-to-hide-your-api-keys-in-python-fb2e1a61b0a0

The following data is described here:

Klisch, A.; Atzberger, C. Operational Drought Monitoring in
Kenya Using MODIS NDVI Time Series. Remote Sens. 2016, 8, 267.
https://www.mdpi.com/2072-4292/8/4/267

For access please email:
clement.atzberger@boku.ac.at

and copy the ftp links to `./etc/conda/activate.d/env_vars.sh`
> #!/bin/sh
> export FTP_1000 = '<ftp link here>'
> export FTP_250 = '<ftp link here>'
"""
import os
from pathlib import Path
import urllib.request
from pathlib import Path
from bs4 import BeautifulSoup
import xarray as xr
import numpy as np
import shutil

from typing import List, Tuple

from .base import BaseExporter
from ..utils import Region, region_lookup

gdal = None


class BokuNDVIExporter(BaseExporter):

    def __init__(self, data_folder: Path = Path("data")):
        super().__init__(data_folder)

        # try and import gdal
        print(
            "The BOKU NDVI exporter requires GDAL"
            "The mac conda environment contains it."
            "In addition, a (finnicky) ubuntu environment contains them in "
            "environment.ubuntu.cpu.gdal.yml"
        )
        global gdal
        if gdal is None:
            from osgeo import gdal

    @staticmethod
    def get_filenames(url: str, identifying_string: str) -> List[str]:
        """ Get the filenames from the ftp url"""
        # use urllib.request to read the page source
        req = urllib.request.Request(url)
        response = urllib.request.urlopen(req)
        the_page = response.read()

        # use BeautifulSoup to parse the html source
        page = str(BeautifulSoup(the_page, features="lxml"))  # type: ignore

        # split the page to get the filenames as a list
        firstsplit = page.split('\r\n')  # split the newlines
        secondsplit = [x.split(' ') for x in firstsplit]  # split the spaces
        flatlist = [
            item for sublist in secondsplit for item in sublist
        ]  # flatten

        # get the name of the files by the identifying_string
        files = [f for f in flatlist if identifying_string in f]
        return files

    @staticmethod
    def wget_file(url_filepath: str, output_folder: Path) -> None:
        """
        https://explainshell.com/explain?cmd=wget+-np+-nH+--cut
        -dirs+7+www.google.come+-P+folder
        """
        os.system(f"wget -np -nH --cut-dirs 2 {url_filepath} \
            -P {output_folder.as_posix()}")

    @staticmethod
    def tif_to_nc(tif_file: Path, nc_file: Path) -> None:
        ds = gdal.Open(tif_file.resolve().as_posix())  # type: ignore
        _ = gdal.Translate(format='NetCDF', srcDS=ds,  # type: ignore
                            destName=nc_file.resolve().as_posix())

    def export(
        self,
        region_name: str = "kenya",
    ) -> None:
        """
        Export BOKU processed MODIS NDVI data

        Arguments
        ----------
        region_name: str = 'kenya'
            The region to download. Must be one of the regions in the
            region_lookup dictionary
        """

        identifying_string = '.tif'

        # 1. download tif files
        fnames = self.get_filenames(self.base_url, identifying_string)
        urls = [self.base_url + f for f in fnames]

        for url in urls:
            self.wget_file(url, output_dir)

        tif_files = [f for f in self.raw_folder.glob('*.tif')]
        tif_files.sort()

        # 2. move tif files to /tif directory
        dst_dir = self.raw_folder / 'tifs'
        if not dst_dir.exists():
            dst_dir.mkdir(exist_ok=True, parents=True)

        dst_tif_files = [dst_dir / f.name for f in tif_files]

        for src, dst in zip(tif_files, dst_tif_files):
            shutil.move(src, dst)

        # 3. convert from tif to netcdf
        tif_files = [f for f in raw_dir.glob('tifs/*.tif')]
        TMP_nc_files = [f.parents[0] / (f.stem + '_TMP.nc') for f in tif_files]
        nc_files = [f.parents[0] / (f.stem + '.nc') for f in tif_files]
        tif_files.sort()
        TMP_nc_files.sort()
        nc_files.sort()

        print('\n')
        for tif_file, nc_file in zip(tif_files, TMP_nc_files):
            self.tif_to_nc(tif_file, nc_file)
            print(f'-- Converted {tif_file.name} to netcdf --')

        # 4. rename BAND1 to rename_str
        rename_str = 'boku_ndvi'

        TMP_nc_files = [f for f in raw_dir.glob('*TMP.nc')]
        nc_files = [f.parents[1] / (f.stem + '.nc') for f in tif_files]

        TMP_nc_files.sort()
        nc_files.sort()

        print('\n')
        for tmp_file, nc_file in zip(TMP_nc_files, nc_files):
            ds = xr.open_dataset(tmp_file).rename(dict(Band1=rename_str))
            da = ds[rename_str]
            da.to_netcdf(nc_file)
            print(f'-- Renamed {nc_file.name} to {rename_str} --')

        # 5. remove temporary netcdf files
        [f.unlink() for f in TMP_nc_files]
        print('Removed *TMP.nc files')




class BokuNDVI1000Exporter(BokuNDVIExporter):
    # 1km pixel
    name: str = 'boku_ndvi_1000'
    resolution: str = '1000'
    base_url: str = os.environ.get("FTP_1000")


class BokuNDVI250Exporter(BokuNDVIExporter):
    # 250m pixel
    name: str = 'boku_ndvi_250'
    resolution: str = '250'
    base_url: str = os.environ.get("FTP_250")
