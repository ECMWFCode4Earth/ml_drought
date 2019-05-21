"""
- Add lat lon coordinates
- add time coordinates
- subset Kenya
- merge into one time (~500MB)
"""
from pathlib import Path
import pathlib
import xarray as xr
import multiprocessing
from typing import List, Optional
import pickle
from functools import partial

from xarray import Dataset

from .base import (BasePreProcessor,)
from .preprocess_vhi import (
    extract_timestamp,
    create_lat_lon_vectors,
    create_new_dataset,
    create_filename,
)
from .preprocess_utils import select_bounding_box_xarray


class CHIRPSPreprocesser(BasePreProcessor):
    """ Preprocesses the CHIRPS data """
