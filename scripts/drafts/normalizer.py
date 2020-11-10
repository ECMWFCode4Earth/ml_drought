from collections import defaultdict
from typing import DefaultDict, Dict, List, Tuple
from pathlib import Path
import xarray as xr


# class Normalizer:
def calculate_normalization_dict(
    ds: xr.Dataset, static: bool = False, global_mean: bool = False
) -> DefaultDict[str, Dict[str, float]]:
    """[summary]

    Args:
        ds (xr.Dataset): [description]
        static (bool, optional): [description]. Defaults to False.

    Returns:
        DefaultDict[str, Dict[str, float]]: [description]
    """
    normalization_values: DefaultDict[str, Dict[str, float]] = defaultdict(dict)
    if static:
        reducing_dims = ["lat", "lon"]
    else:
        reducing_dims = ["lat", "lon", "time"]

    for var in ds.data_vars:
        if var.endswith("one_hot"):
            #  Should not normalize one_hot encoded variables
            mean = None
            std = None
        else:
            mean = float(ds[var].mean(dim=reducing_dims, skipna=True).values)
            std = float(ds[var].std(dim=reducing_dims, skipna=True).values)

        # ensure that std DOES NOT EQUAL Zero
        if std <= 0:
            std = 1e-10

        normalization_values[var]["mean"] = mean
        normalization_values[var]["std"] = std

    return normalization_values


def normalize_xr(
    ds: xr.Dataset, static: bool = False
) -> Tuple[xr.Dataset, DefaultDict[str, Dict[str, float]]]:
    """Normalize the xarray object

    Args:
        ds (xr.Dataset): [description]
        static (bool, optional): [description]. Defaults to False.

    Returns:
        xr.Dataset: Xarray Dataset with normalized values
        DefaultDict[str, Dict[str, float]]: The values
    """
    norm_dict = calculate_normalization_dict(ds, static)

    list_of_normed: List[xr.DataArray] = []
    for variable in ds.data_vars:
        list_of_normed.append(
            (ds[variable] - norm_dict[variable]["mean"]) / norm_dict[variable]["std"]
        )

    ds_norm = xr.merge(list_of_normed)

    return ds_norm, norm_dict


def unnormalize_xr(
    ds: xr.Dataset, normalization_dict: DefaultDict[str, Dict[str, float]]
) -> xr.Dataset:
    list_of_unnormed: List[xr.DataArray] = []
    for variable in ds.data_vars:
        list_of_unnormed.append(
            (ds[variable] * normalization_dict[variable]["std"])
            + normalization_dict[variable]["mean"]
        )

    ds_unnorm = xr.merge(list_of_unnormed)

    return ds_unnorm
