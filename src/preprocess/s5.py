from pathlib import Path
from functools import partial
import xarray as xr
import multiprocessing
from shutil import rmtree
from typing import Optional

from .base import BasePreProcessor

class S5Preprocessor(BasePreProcessor):

    dataset = 's5'


    ds = xr.open_dataset('era5-levels-members.grib', engine='cfgrib')
