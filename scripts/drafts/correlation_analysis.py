from scipy.stats import spearmanr
from collections import defaultdict
import xarray as xr
from pathlib import Path


if __name__ == "__main__":
    data_dir = Path("/cats/datastore/data")
    all_preds = xr.open_dataset(data_dir / "RUNOFF/all_preds.nc")
    pass
