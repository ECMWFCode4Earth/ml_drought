import cdsapi
from pathlib import Path
import certifi
import urllib3
import warnings
import itertools
import re
from pprint import pprint

from typing import Dict, Optional, List

from .base import BaseExporter, Region, get_kenya
from .cds import CDSExporter

http = urllib3.PoolManager(
    cert_reqs='CERT_REQUIRED',
    ca_certs=certifi.where()

class S5Exporter(CDSExporter):

    def __init__():
        pass

    pass
    
