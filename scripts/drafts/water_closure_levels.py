from typing import List, Tuple, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

import sys
sys.path.append("../../")
from scripts.utils import get_data_path


def assign_wateryear(dt):
    """https://stackoverflow.com/a/52615358/9940782"""
    dt = pd.Timestamp(dt)
    if dt.month >= 10:
        return(pd.datetime(dt.year+1, 1, 1).year)
    else:
        return(pd.datetime(dt.year, 1, 1).year)


def create_closure_masks(
    ds: xr.Dataset,
    closure_thresholds: List[float] = [0.05, 0.1, 0.15, 0.2, 0.25],
    method: str = "sum",
) -> Tuple[xr.Dataset]:
    assert all([(c <= 1) & (c > 0) for c in closure_thresholds]), "Need to be between 0 and 1 (ratio)"
    # assign water year as a coordinate
    ds_wy = ds.assign_coords(wy=("time", [assign_wateryear(dt) for dt in ds.time.values]))

    # calculate water balance and mean precip
    if method == "sum":
        ann_aggregated = ds_wy.groupby("wy").sum().isel(wy=slice(1, -1))
    elif method == "mean":
        ann_aggregated = ds_wy.groupby("wy").mean().isel(wy=slice(1, -1))
    else:
        assert False, "One of [sum, mean] required for calculating WaterYear WB"

    ann_closure = ann_aggregated["precipitation"] - ann_aggregated["pet"] - ann_aggregated["discharge_spec"]
    mean_closure = ann_closure.mean('wy')
    mean_precip = ann_aggregated["precipitation"].mean("wy")

    # calculate outside the error
    masks = []
    for c in closure_thresholds:
        # calculate the % of the precip marking the 'error'
        thresh = mean_precip * c
        ref_value_upper = 0.5 * thresh
        ref_value_lower = -0.5 * thresh

        bool_wb = ((ref_value_lower < mean_closure) & (ref_value_upper > mean_closure))
        bool_wb = bool_wb.to_dataset(**dict(name="mask")).assign_coords(threshold=c).expand_dims(dim="threshold")
        masks.append(bool_wb)

    threshold = xr.concat(masks, dim='threshold')

    return threshold, (mean_closure, mean_precip)


def get_condition_sids(thresholds: xr.Dataset, threshold_level: float):
    assert threshold_level in thresholds.threshold.values, f"Threshold level must be in: {thresholds.threshold.values}"
    condition_sids = thresholds.where(thresholds.sel(threshold=threshold_level)["mask"], drop=True).station_id.values
    return condition_sids


if __name__ == "__main__":
    data_dir = Path("/cats/datastore/data")

    ds = xr.open_dataset(data_dir / "RUNOFF/ALL_dynamic_ds.nc")
    ds['station_id'] = ds['station_id'].astype(int)

    # calculate thresholds
    thresholds, (mean_closure, mean_precip) = create_closure_masks(ds, method="sum", closure_thresholds=[0.05, 0.1, 0.15, 0.2, 0.25, 1.0])
