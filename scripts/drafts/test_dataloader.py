from pathlib import PosixPath
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CamelsDataloader(Dataset):
    def __init__(self, static: xr.Dataset, dynamic: xr.Dataset, seq_length: int = 365, mode: str = "train"):
        self.seq_length = seq_length
        self.mode = mode

        # placeholder to store std of discharge, used for rescaling losses during training
        self.q_std = None

        self.x, self.y = self._load_data()
        self.num_samples = self.x.shape[0]


    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self.static = static
        self.dynamic = dynamic

    def __getitem__(self, index):

    def __len__(self):
        return self.num_samples


class CamelsH5(Dataset):
    """PyTorch data set to work with pre-packed hdf5 data base files.
    Should be used only in combination with the files processed from `create_h5_files` in the
    `papercode.utils` module.
    Parameters
    ----------
    h5_file : PosixPath
        Path to hdf5 file, containing the bundled data
    basins : List
        List containing the 8-digit USGS gauge id
    db_path : str
        Path to sqlite3 database file, containing the catchment characteristics
    concat_static : bool
        If true, adds catchment characteristics at each time step to the meteorological forcing
        input data, by default False
    cache : bool, optional
        If True, loads the entire data into memory, by default False
    no_static : bool, optional
        If True, no catchment attributes are added to the inputs, by default False
    """

    def __init__(self,
                 dynamic_path: PosixPath,
                 static_path: PosixPath,
                 basins: List,
                 concat_static: bool = False,
                 cache: bool = True,
                 no_static: bool = False):
        self.dynamic_path = dynamic_path
        self.static_path = static_path
        self.basins = basins
        self.concat_static = concat_static
        self.cache = cache
        self.no_static = no_static

        # Placeholder for catchment attributes stats
        self.df = None
        self.attribute_means = None
        self.attribute_stds = None
        self.attribute_names = None

        # preload data if cached is true
        if self.cache:
            (self.x, self.y, self.sample_2_basin, self.q_stds) = self._preload_data()

        # load attributes into data frame
        self._load_attributes()

        # determine number of samples once
        if self.cache:
            self.num_samples = self.y.shape[0]
        else:
            with h5py.File(h5_file, 'r') as f:
            with h5py.File(h5_file, 'r') as f:
                self.num_samples = f["target_data"].shape[0]

    # -----------------------------------
    # IMPORTANT FUNCTIONS
    # -----------------------------------
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        # ALWAYS cache data
        x = self.x[idx]
        y = self.y[idx]
        basin = self.sample_2_basin[idx]
        q_std = self.q_stds[idx]

        # work with static data
        if not self.no_static:
            # get attributes from data frame and create 2d array with copies
            attributes = self.df.loc[self.df.index == basin].values

            if self.concat_static:
                attributes = np.repeat(attributes, repeats=x.shape[0], axis=0)
                # combine meteorological obs with static attributes
                x = np.concatenate([x, attributes], axis=1).astype(np.float32)
            else:
                attributes = torch.from_numpy(attributes.astype(np.float32))

        # convert to torch tensors
        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32))
        q_std = torch.from_numpy(q_std)

        if self.no_static:
            return x, y, q_std
        else:
            if self.concat_static:
                return x, y, q_std
            else:
                return x, attributes, y, q_std

    # -----------------------------------
    # Other Functions
    # -----------------------------------
    def _preload_data(self):
        with h5py.File(self.h5_file, 'r') as f:
            x = f["input_data"][:]
            y = f["target_data"][:]
            str_arr = f["sample_2_basin"][:]
            str_arr = [x.decode("ascii") for x in str_arr]
            q_stds = f["q_stds"][:]
        return x, y, str_arr, q_stds

    def _get_basins(self):
        if self.cache:
            basins = list(set(self.sample_2_basin))
        else:
            with h5py.File(self.h5_file, 'r') as f:
                str_arr = f["sample_2_basin"][:]
            str_arr = [x.decode("ascii") for x in str_arr]
            basins = list(set(str_arr))
        return basins

    def _load_attributes(self):
        df = load_attributes(self.db_path, self.basins, drop_lat_lon=True)

        # store means and stds
        self.attribute_means = df.mean()
        self.attribute_stds = df.std()

        # normalize data
        df = (df - self.attribute_means) / self.attribute_stds

        self.attribute_names = df.columns
        self.df = df

    def get_attribute_means(self) -> pd.Series:
        """Return means of catchment attributes

        Returns
        -------
        pd.Series
            Contains the means of each catchment attribute
        """
        return self.attribute_means

    def get_attribute_stds(self) -> pd.Series:
        """Return standard deviation of catchment attributes

        Returns
        -------
        pd.Series
            Contains the stds of each catchment attribute
        """
        return self.attribute_stds
