import xarray as xr
from pathlib import Path
# import mapclassify as mc
from typing import Optional


class BaseIndices:

    name: str  # implemented by child classes
    index: xr.Dataset
    ds: xr.Dataset
    resample: bool = False

    def __init__(self,
                 data_path: Path,
                 resample_str: Optional[str] = None) -> None:
        """
        Arguments:
        ---------
        data_path: Path

        resample_str: Optional[str]
            One of {'daysofyear', 'month', 'year', 'season', None}
        """
        self.data_path = data_path
        assert self.data_path.exists(), f"{self.data_path} does not exist.\
        Must be directed to an existing .nc file!"

        self.ds = xr.open_dataset(data_path)

        if resample_str is not None:
            self.resample = True
            self.resample_str = resample_str
            self.ds = self.resample_ds_mean()

    def resample_ds_mean(self) -> xr.Dataset:
        lookup = {
            'month': 'M',
            'year': 'Y',
            'season': 'Q-DEC',
            'daysofyear': 'D',
            None: None,
        }
        return self.ds.resample(time=f'{lookup[self.resample_str]}').mean()

    def save(self):
        """save the self.index to netcdf"""
        raise NotImplementedError

    @staticmethod
    def rolling_cumsum(ds: xr.Dataset, rolling_window=3):
        pass
