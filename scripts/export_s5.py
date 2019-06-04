from src.exporters.s5 import (S5Exporter)
from pathlib import Path
import numpy as np

# %load_ext autoreload
# %autoreload 2

data_dir = Path('data')

granularity = 'hourly'
pressure_level=False

s5 = S5Exporter(
    data_folder=data_dir,
    granularity=granularity,
    pressure_level=pressure_level,
)

variable = 'total_precipitation'
min_year = 2015
max_year = 2018
min_month = 1
max_month = 12
max_leadtime = None
pressure_levels = [200, 500, 925]
selection_request = None
N_parallel_requests = 20
show_api_request = True

s5.export(
    variable=variable,
    min_year=min_year,
    max_year=max_year,
    min_month=min_month,
    max_month=max_month,
    max_leadtime=max_leadtime,
    pressure_levels=pressure_levels,
    N_parallel_requests=N_parallel_requests,
)
