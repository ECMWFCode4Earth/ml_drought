import xarray as xr
from pathlib import Path
from typing import List

data_dir = Path("/cats/datastore/data")
save_dir = data_dir / "000_models/"


def _read_xarray_objs(all_files: List[Path]) -> xr.Dataset:
    _ds = [xr.open_dataset(f) for f in all_files]
    ds = xr.concat(_ds, dim="time")
    ds = ds.sortby("time")

    return ds


# READ PREDICTIONS
for target_var in ["VCI1M", "VCI3M"]:
    all_ds = []
    for model in [
        d.name
        for d in (
            data_dir / f"000_models/models_{target_var}/one_month_forecast/"
        ).iterdir()
    ]:
        ds = xr.concat(
            [
                xr.open_dataset(f)
                for f in (
                    data_dir
                    / f"000_models/models_{target_var}/one_month_forecast/{model}"
                ).glob("*.nc")
            ],
            dim="time",
        )
        ds = ds.sortby("time").rename({"preds": model})
        all_ds.append(ds)

    all_ds = xr.merge(all_ds)
    all_ds.to_netcdf(save_dir / f"{target_var}_predictions.nc")

#  READ TRAINING DATA
save_dir = data_dir / "000_features/"

all_fs = [f for f in (data_dir / "000_features/one_month_forecast/test").glob("*/x.nc")]
X_test = _read_xarray_objs(all_fs)
all_fs = [f for f in (data_dir / "000_features/one_month_forecast/test").glob("*/y.nc")]
y_test = _read_xarray_objs(all_fs)

all_fs = [
    f for f in (data_dir / "000_features/one_month_forecast/train").glob("*/x.nc")
]
X_train = _read_xarray_objs(all_fs)
all_fs = [
    f for f in (data_dir / "000_features/one_month_forecast/train").glob("*/y.nc")
]
y_train = _read_xarray_objs(all_fs)

X_test.to_netcdf(save_dir / "X_test.nc")
y_test.to_netcdf(save_dir / "y_test.nc")
X_train.to_netcdf(save_dir / "X_train.nc")
y_train.to_netcdf(save_dir / "y_train.nc")
