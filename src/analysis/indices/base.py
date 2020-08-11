import xarray as xr
from pathlib import Path
from typing import Optional


class BaseIndices:

    name: str  # implemented by child classes
    index: xr.Dataset
    ds: xr.Dataset
    resample: bool = False

    def __init__(
        self,
        file_path: Optional[Path] = None,
        ds: Optional[xr.Dataset] = None,
        resample_str: Optional[str] = None,
    ) -> None:
        """
        Arguments:
        ---------
        file_path: Path
            Path to the `.nc` file that you want to use for calculating
             indices from.

        resample_str: Optional[str]
            One of {'daysofyear', 'month', 'year', 'season', None}
        """
        assert (file_path is not None) or (
            ds is not None
        ), "Either file_path or ds must be provided"

        if file_path is not None:
            self.file_path = file_path
            assert (
                self.file_path.exists()
            ), f"{self.file_path} does not exist.\
            Must be directed to an existing .nc file!"

            self.ds = xr.open_dataset(file_path)
        elif ds is not None:
            self.ds = ds
        else:
            assert False, "Must provide ds or file_path argument"

        if resample_str is not None:
            self.resample = True
            self.resample_str = resample_str
            self.ds = self.ds.sortby("time")
            self.ds = self.resample_ds_mean()

    def resample_ds_mean(self) -> xr.Dataset:
        lookup = {
            "month": "M",
            "M": "M",
            "year": "Y",
            "Y": "Y",
            "season": "Q-DEC",
            "Q-DEC": "Q-DEC",
            "daysofyear": "D",
            "daily": "D",
            "days": "D",
            "D": "D",
            None: None,
        }
        assert self.resample_str in [
            k for k in lookup.keys()
        ], f"resample_str must be one of: {[k for k in lookup.keys()]}"
        return self.ds.resample(time=f"{lookup[self.resample_str]}").mean()

    def save(self, data_dir: Path = Path("data")):
        """save the self.index to netcdf"""
        analysis_dir = data_dir / "analysis" / "indices"
        if not analysis_dir.exists():
            analysis_dir.mkdir(parents=True, exist_ok=True)

        file_path = analysis_dir / (self.name + ".nc")
        self.index.to_netcdf(file_path)
        print(f"Saved {self.name} to {file_path.as_posix()}")
