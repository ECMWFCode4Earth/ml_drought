from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
from typing import Dict, List, Tuple, Union, Optional
import pickle
import tqdm
from torch.utils.data import Dataset
import torch
import h5py
import sys
from .runoff_utils import (
    load_static_data,
    get_basins,
    reshape_data,
    CalculateNormalizationParams,
)
from .basin import CamelsCSV
from src.utils import _rename_directory


class RunoffEngineer:
    """Class to write the data to data/features/features.h5 file for training.

    Functions:
        `self.create_training_data()`

    Attrs:
        data_dir (Path):
            The data directory
        basins (List[int]):
            List of basin IDs
        train_dates (List[int]):
            List of the years used for training
        with_basin_str (bool, optional):
            Whether to save the basin ID associated with each datapoint. Defaults to True.
        target_var (str, optional):
            The target variable (y). Defaults to "discharge_spec".
        x_variables (Optional[List[str]], optional):
            The x variables (x_dynamic) included in the model. Defaults to ["precipitation", "peti"].
        static_variables (Optional[List[str]], optional):
            The static variables (x_static). Defaults to None.
        ignore_static_vars (Optional[List[str]], optional):
            List of static variables to ignore. Defaults to None.
        seq_length (int, optional):
            The number of previous timesteps included as input to the model. Defaults to 365.
        with_static (bool, optional):
            Whether to include static data in the model. Defaults to False.
        concat_static (bool, optional):
            Whether to concatenate static data or not. Defaults to False.

    Raises:
        FileExistsError: If the h5 file is already created, raise a
            file exists error.
    """

    def __init__(
        self,
        data_dir: Path,
        basins: List[int],
        train_dates: List[int],
        with_basin_str: bool = True,
        target_var: str = "discharge_spec",
        x_variables: Optional[List[str]] = ["precipitation", "peti"],
        static_variables: Optional[List[str]] = None,
        ignore_static_vars: Optional[List[str]] = None,
        seq_length: int = 365,
        with_static: bool = True,
        concat_static: bool = False,
    ):
        self.data_dir = data_dir
        self.out_file = self.data_dir / "features/features.h5"

        if not self.out_file.parents[0].exists():
            self.out_file.parents[0].mkdir(exist_ok=True, parents=True)

        if self.out_file.is_file():
            print(f"File already exists at {self.out_file}")
            # move file
            _rename_directory(self.out_file, self.out_file, with_datetime=True)

        # Experiment Data Params
        self.train_dates = train_dates
        self.target_var = target_var
        self.x_variables = x_variables
        self.static_variables = static_variables
        self.seq_length = seq_length
        self.with_static = with_static
        self.concat_static = concat_static
        self.basins = basins
        self.with_basin_str = with_basin_str

        # derived params
        if x_variables is not None:
            self.n_variables = len(x_variables)

        # normalisation dictionary
        self.normalization_dict = CalculateNormalizationParams(
            data_dir=self.data_dir,
            train_dates=self.train_dates,
            target_var=self.target_var,
            x_variables=self.x_variables,
            static_variables=self.static_variables,
        ).normalization_dict

        # save normalization dict
        pickle.dump(
            self.normalization_dict,
            open(self.out_file.parents[0] / "normalization_dict.pkl", "wb"),
        )

    def create_training_data(self):
        with h5py.File(self.out_file, "w") as out_f:
            input_data = out_f.create_dataset(
                "input_data",
                shape=(0, self.seq_length, self.n_variables),
                maxshape=(None, self.seq_length, self.n_variables),
                chunks=True,
                dtype=np.float32,
                compression="gzip",
            )
            target_data = out_f.create_dataset(
                "target_data",
                shape=(0, 1),
                maxshape=(None, 1),
                chunks=True,
                dtype=np.float32,
                compression="gzip",
            )

            target_stds = out_f.create_dataset(
                "target_stds",
                shape=(0, 1),
                maxshape=(None, 1),
                dtype=np.float32,
                compression="gzip",
                chunks=True,
            )

            if self.with_basin_str:
                sample_2_basin = out_f.create_dataset(
                    "sample_2_basin",
                    shape=(0,),
                    maxshape=(None,),
                    dtype="S10",
                    compression="gzip",
                    chunks=True,
                )

            for basin in tqdm.tqdm(self.basins, file=sys.stdout):
                dataset = CamelsCSV(
                    data_dir=self.data_dir,
                    basin=basin,
                    train_dates=self.train_dates,
                    normalization_dict=self.normalization_dict,
                    is_train=True,
                    target_var=self.target_var,
                    x_variables=self.x_variables,
                    static_variables=self.static_variables,
                    seq_length=self.seq_length,
                    with_static=self.with_static,
                    concat_static=self.concat_static,
                )

                num_samples = len(dataset)
                if num_samples < 1:
                    print(f"No data for basin: {basin}. Skipping ...")
                    continue
                total_samples = input_data.shape[0] + num_samples

                # store input / output samples
                input_data.resize((total_samples, self.seq_length, self.n_variables))
                target_data.resize((total_samples, 1))
                # (append to already existing in h5 files)
                input_data[-num_samples:, :, :] = dataset.x
                target_data[-num_samples:, :] = dataset.y

                # store std of discharge of this basin for each sample
                target_stds.resize((total_samples, 1))
                target_std_array = np.array(
                    [dataset.target_std] * num_samples, dtype=np.float32
                ).reshape(-1, 1)
                target_stds[-num_samples:, :] = target_std_array

                # store the basin id as a string
                if self.with_basin_str:
                    sample_2_basin.resize((total_samples,))
                    str_arr = np.array(
                        [str(basin).encode("ascii", "ignore")] * num_samples
                    )
                    sample_2_basin[-num_samples:] = str_arr

                out_f.flush()


class CamelsDataLoader(Dataset):
    """PyTorch data set to work with pre-packed hdf5 data base files.

    Should be used only in combination with the files processed from `create_h5_files` in the
    `papercode.utils` module.

    Parameters
    ----------
    h5_file : Path
        Path to hdf5 file, containing the bundled data
    basins : List
        List containing the 8-digit USGS gauge id
    static_data_path : str
        path to static .nc file
    concat_static : bool
        If true, adds catchment characteristics at each time step to the meteorological forcing
        input data, by default False
    cache : bool, optional
        If True, loads the entire data into memory, by default False
    with_static : bool, optional
        If True, no catchment attributes are added to the inputs, by default False
    """

    def __init__(
        self,
        data_dir: Path,
        basins: List[str],
        train_dates: List[int],
        target_var: str = "discharge_spec",
        concat_static: bool = False,
        cache: bool = False,
        with_static: bool = False,
    ):
        # Paths to data
        self.data_dir = data_dir
        self.h5_file = data_dir / "features/features.h5"
        self.static_data_path = data_dir / "interim/static/data.nc"

        assert self.h5_file.exists(), "Has the `RunoffEngineer` been run?"
        assert (
            self.static_data_path.exists()
        ), "Has the `CAMELSGBPreprocessor` been run?"

        # Experiment params
        self.basins = basins
        self.concat_static = concat_static
        self.cache = cache
        self.with_static = with_static
        self.train_dates = train_dates

        self.normalization_dict = pickle.load(
            open(self.h5_file.parents[0] / "normalization_dict.pkl", "rb")
        )
        self.target_var = self.normalization_dict["target_var"]
        self.x_variables = self.normalization_dict["x_variables"]
        self.static_variables = self.normalization_dict["static_variables"]

        # Placeholder for catchment attributes stats
        self.static_df = None
        self.attribute_means = None
        self.attribute_stds = None
        self.attribute_names = None

        # preload data if cached is true
        if self.cache:
            (
                self.x,
                self.y,
                self.sample_2_basin,
                self.target_stds,
            ) = self._preload_data()

        # load attributes into data frame
        self.static_df = self._load_static_data()

        # determine number of samples once
        if self.cache:
            self.num_samples = self.y.shape[0]
        else:
            with h5py.File(self.h5_file, "r") as f:
                self.num_samples = f["target_data"].shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        if self.cache:
            x = self.x[idx]
            y = self.y[idx]
            basin = self.sample_2_basin[idx]
            target_std = self.target_stds[idx]

        else:
            with h5py.File(self.h5_file, "r") as f:
                x = f["input_data"][idx]
                y = f["target_data"][idx]
                basin = f["sample_2_basin"][idx]
                basin = basin.decode("ascii")
                target_std = f["target_stds"][idx]

        if self.with_static:
            # get attributes from data frame and create 2d array with copies
            attributes = self.static_df.loc[self.static_df.index == int(basin)].values

            if self.concat_static:
                attributes = np.repeat(attributes, repeats=x.shape[0], axis=0)
                # combine meteorological obs with static attributes
                x = np.concatenate([x, attributes], axis=1).astype(np.float32)
            else:
                attributes = torch.from_numpy(attributes.astype(np.float32))

        # convert to torch tensors
        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32))
        target_std = torch.from_numpy(target_std)

        if self.with_static:
            if self.concat_static:
                return x, y, target_std
            else:
                return x, attributes, y, target_std
        else:
            return x, y, target_std

    def _preload_data(
        self
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with h5py.File(self.h5_file, "r") as f:
            x = f["input_data"][:]
            y = f["target_data"][:]
            str_arr = f["sample_2_basin"][:]
            str_arr = [x.decode("ascii") for x in str_arr]
            target_stds = f["target_stds"][:]

        return x, y, str_arr, target_stds

    def _get_basins(self) -> List[str]:
        """Return list of basins

        Returns:
            List[str]:
        """
        if self.cache:
            basins: List[str] = list(set(self.sample_2_basin))
        else:
            with h5py.File(self.h5_file, "r") as f:
                str_arr = f["sample_2_basin"][:]
            str_arr = [x.decode("ascii") for x in str_arr]
            basins = list(set(str_arr))
        return basins

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

    def _load_static_data(self) -> pd.DataFrame:
        """Return the static data as an pd.DataFrame, read from the
        xr.Dataset (netcdf file).

        Returns:
            pd.DataFrame: Static catchment attributes (static over time)
        """
        static_df = load_static_data(
            data_dir=self.data_dir,
            static_variables=self.static_variables,
            drop_lat_lon=False,
            basins=self.basins,
        )

        # store the means and stds
        self.attribute_means = static_df.mean()
        self.attribute_stds = static_df.std()
        self.attribute_names = static_df.columns

        # normalise the data
        static_df = (
            static_df - self.normalization_dict["static_means"]
        ) / self.normalization_dict["static_stds"]

        return static_df
