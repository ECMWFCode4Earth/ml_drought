""" 
ipython -c 'import xarray as xr; import matplotlib.pyplot as plt; ds = xr.open_dataset("one_time.nc"); ds[[v for v in ds.data_vars][0]].plot(); plt.show()'
"""

import xarray as xr 
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Plot xarray file")
    parser.add_argument(
        "--filepath",
        required=True,
        help="path to netcdf file to plot",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    ds = xr.open_dataset(args.filepath)
    pass