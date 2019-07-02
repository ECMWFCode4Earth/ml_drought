import xarray as xr
import pandas as pd
from pathlib import Path

from typing import List, Dict, Optional, Union, Tuple
from enum import Enum

import climate_indices
from climate_indices import indices
from climate_indices.__main__ import _spi

from .base import BaseIndices


class ZScoreIndex(BaseIndices):
    """https://climatedataguide.ucar.edu/climate-data/standardized-precipitation-index-spi"""

    name = 'z_score_index'
