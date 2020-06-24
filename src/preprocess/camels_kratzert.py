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


# CAMELS catchment characteristics ignored in this study
INVALID_ATTR = [
    "gauge_name",
    "area_geospa_fabric",
    "geol_1st_class",
    "glim_1st_class_frac",
    "geol_2nd_class",
    "glim_2nd_class_frac",
    "dom_land_cover_frac",
    "dom_land_cover",
    "high_prec_timing",
    "low_prec_timing",
    "huc",
    "q_mean",
    "runoff_ratio",
    "stream_elas",
    "slope_fdc",
    "baseflow_index",
    "hfd_mean",
    "q5",
    "q95",
    "high_q_freq",
    "high_q_dur",
    "low_q_freq",
    "low_q_dur",
    "zero_q_freq",
    "geol_porostiy",
    "root_depth_50",
    "root_depth_99",
    "organic_frac",
    "water_frac",
    "other_frac",
]


class CalculateNormalizationParams:
    def __init__(
        self,
        train_dates: List[int],
        x_variables: Optional[List[str]],
        static_variables: Optional[List[str]],
        target_var: str,
        data_dir: Path = Path("data"),
    ):
        self.data_dir = data_dir
        self.train_dates = np.sort(train_dates)
        self.x_variables = x_variables
        self.static_variables = static_variables
        self.target_var = target_var
        assert isinstance(self.target_var, str), "Expect ONE target_var"

        assert (self.data_dir / "interim/static/data.nc").exists()
        assert (self.data_dir / "interim/camels_preprocessed/data.nc").exists()
        dynamic_ds = xr.open_dataset(
            self.data_dir / "interim/camels_preprocessed/data.nc"
        )
        static_ds = xr.open_dataset(self.data_dir / "interim/static/data.nc")

        self.normalization_dict = self.calculate_normalization_dict(
            dynamic_ds=dynamic_ds,
            static_ds=static_ds,
            train_dates=self.train_dates,
            x_variables=self.x_variables,
            static_variables=self.static_variables,
            target_var=self.target_var,
        )

    @staticmethod
    def calculate_normalization_dict(
        dynamic_ds: xr.Dataset,
        static_ds: xr.Dataset,
        train_dates: List[int],
        x_variables: Optional[List[str]],
        static_variables: Optional[List[str]],
        target_var: str,
    ) -> Dict[str, float]:
        """Calculate normalisation parameters (mean and std) over all basins

        Args:
            dynamic_ds (xr.Dataset): dynamic (forcings) Dataset
            static_ds (xr.Dataset): static (attributes) Dataset
            train_dates (List[int]): Training dates to limit std/means to training times
            x_variables (List[str]): list of variables to be included in the model

        Returns:
            Dict[str, float]: Normalisation dictionary
                Keys: [ "static_means", "static_stds",
                    "dynamic_means", "dynamic_stds", "target_mean",
                    "target_std", "x_variables", "target_var",
                    "static_variables", ]
        """
        print("Calculating mean and std for variables")

        dynamic_means = (
            dynamic_ds[
                x_variables
                if x_variables is not None
                else [v for v in dynamic_ds.data_vars if v != target_var]
            ]
            .sel(time=slice(str(train_dates[0]), str(train_dates[-1])))  # type: ignore
            .mean()
            .to_array()
            .values
        )
        dynamic_stds = (
            dynamic_ds[
                x_variables
                if x_variables is not None
                else [v for v in dynamic_ds.data_vars if v != target_var]
            ]
            .sel(time=slice(str(train_dates[0]), str(train_dates[-1])))  # type: ignore
            .std()
            .to_array()
            .values
        )
        target_mean = (
            dynamic_ds[target_var]
            .sel(time=slice(str(train_dates[0]), str(train_dates[-1])))  # type: ignore
            .mean()
            .values
        )
        target_std = (
            dynamic_ds[target_var]
            .sel(time=slice(str(train_dates[0]), str(train_dates[-1])))  # type: ignore
            .std()
            .values
        )

        n_variables = len(
            x_variables
            if x_variables is not None
            else [v for v in dynamic_ds.data_vars if v != target_var]
        )
        assert target_std.shape == (), "Should be scalar std"
        assert (
            len(dynamic_means) == n_variables
        ), "Should be list of length len(x_variables)"

        static_means = static_ds[static_variables].mean().to_array().values

        static_stds = static_ds[static_variables].std().to_array().values

        normalization_dict = {
            "static_means": static_means,
            "static_stds": static_stds,
            "dynamic_means": dynamic_means,
            "dynamic_stds": dynamic_stds,
            "target_mean": target_mean,
            "target_std": target_std,
            "x_variables": x_variables,
            "target_var": target_var,
            "static_variables": static_variables,
        }

        return normalization_dict


def get_basins(data_dir: Path) -> List[int]:
    timeseries_dir = data_dir / "raw/CAMELS_GB_DATASET/Catchment_Timeseries"
    ts_csvs = [d for d in timeseries_dir.glob("*.csv")]
    gauge_ids = [
        int(d.name.split("ies_")[-1].split("_")[0])
        for d in timeseries_dir.glob("*.csv")
    ]
    return gauge_ids


# @njit
def reshape_data(
    x: np.ndarray, y: np.ndarray, seq_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Reshape data into LSTM many-to-one input samples

    Parameters
    ----------
    x : np.ndarray
        Input features of shape [num_samples, num_features]
    y : np.ndarray
        Output feature of shape [num_samples, 1]
    seq_length : int
        Length of the requested input sequences.

    Returns
    -------
    x_new: np.ndarray
        Reshaped input features of shape [num_samples*, seq_length, num_features], where
        num_samples* is equal to num_samples - seq_length + 1, due to the need of a warm start at
        the beginning
    y_new: np.ndarray
        The target value for each sample in x_new
    """
    # (n_times, n_features)
    num_samples, num_features = x.shape

    # (n_times, seq_length, n_features)
    x_new = np.zeros((num_samples - seq_length + 1, seq_length, num_features))
    # (n_times, 1)
    y_new = np.zeros((num_samples - seq_length + 1, 1))

    # fill in the values
    for i in range(0, x_new.shape[0]):
        x_new[i, :, :num_features] = x[i : i + seq_length, :]
        y_new[i, :] = y[i + seq_length - 1, 0]

    return x_new, y_new


def _prepare_data(data_dir: Path, basins: List[str]):
    db_path = data_dir / "features/static/attributes.db"


class CamelsCSV(Dataset):
    def __init__(
        self,
        basin: str,
        train_dates: List[int],  # dates
        normalization_dict: Dict[str, float],
        data_dir: Path = Path("data"),
        is_train: bool = True,
        target_var: str = "discharge_spec",
        x_variables: Optional[List[str]] = ["precipitation", "peti"],
        static_variables: Optional[List[str]] = None,
        ignore_static_vars: Optional[List[str]] = None,
        seq_length: int = 365,
        with_static: bool = False,
        concat_static: bool = False,
    ):
        self.data_dir = data_dir
        self.normalization_dict = normalization_dict

        # initialise paths
        base_camels_dir = self.data_dir / "raw/CAMELS_GB_DATASET"
        self.attributes_dir = base_camels_dir / "Catchment_Attributes"
        self.timeseries_dir = base_camels_dir / "Catchment_Timeseries"
        self.shp_path = (
            base_camels_dir / "Catchment_Boundaries/CAMELS_GB_catchment_boundaries.shp"
        )
        assert self.attributes_dir.exists()
        assert self.timeseries_dir.exists()

        # data globals
        self.seq_length = seq_length
        self.target_var = target_var
        self.x_variables = x_variables
        self.static_variables = static_variables
        self.ignore_static_vars = ignore_static_vars

        # iteration params
        self.basin = basin
        self.is_train = is_train
        self.train_dates = np.sort(train_dates)
        assert all(
            [
                (isinstance(date, int) or (isinstance(int(date), int)))
                for date in train_dates
            ]
        ), "train_dates must be an array of integers (the years to be used)"

        # means and stds
        self.with_static = with_static
        self.concat_static = concat_static

        # get static data (preprocess if not preprocessed)
        # get all filepaths for csv files in CAMELS_GB_DATASET folder
        self.ts_csvs = [d for d in self.timeseries_dir.glob("*.csv")]
        self.gauge_ids = [
            int(d.name.split("ies_")[-1].split("_")[0])
            for d in self.timeseries_dir.glob("*.csv")
        ]

        # placeholder to store std of discharge, used for rescaling losses during training
        self.q_std = None

        # placeholder to store start and end date of entire period (incl warmup)
        self.period_start = None
        self.period_end = None
        self.attribute_names = None

        self.x, self.y = self._load_dynamic_data()

        if self.with_static:
            self.attributes = self._load_static_data()

        # NUM timesteps
        self.num_samples = self.x.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(
        self, idx: int
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """iterate through the timesteps for each station
        returning X_dyn, X_stat, y.

        Args:
            idx (int): timestep number

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: X, y
            OR
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: X_dyn, X_stat y
        """
        if self.with_static:
            if self.concat_static:  # return concat(dynamic, static), target
                x = torch.cat(
                    [self.x[idx], self.attributes.repeat((self.seq_length, 1))], dim=-1
                )
                return x, self.y[idx]
            else:  # return dynamic, static, target
                return self.x[idx], self.attributes, self.y[idx]
        else:  # return only dynamic / target
            return self.x[idx], self.y[idx]

    def _get_one_basin_csv_file(self) -> Path:
        # get gauge_id
        gauge_ids = [
            d.name.split("ies_")[-1].split("_")[0]
            for d in self.timeseries_dir.glob("*.csv")
        ]
        bool_list = [str(id) == str(self.basin) for id in gauge_ids]
        assert (
            sum(bool_list) == 1
        ), f"Only expect to find one gauge id with id : {self.basin}"

        csv_file = np.array([d for d in self.timeseries_dir.glob("*.csv")])[bool_list][
            0
        ]

        return csv_file

    def _load_basin_dynamic_data(self):
        # load a single basin!
        csv_file = self._get_one_basin_csv_file()
        df = pd.read_csv(csv_file)

        # set datatypes
        df["date"] = df["date"].astype(np.dtype("datetime64[ns]"))
        df = df.set_index("date")
        df = df.astype("float")

        return df

    def _get_dynamic_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the dynamic X variables and the target variable (y)
        """
        basin_df = self._load_basin_dynamic_data()

        # get the features of interest
        if self.x_variables is not None:
            basin_df = basin_df[self.x_variables + [self.target_var]]

        # get the time series of interest
        basin_df = self._get_ts_of_interest(basin_df)

        self.period_start = basin_df.index[0]
        self.period_end = basin_df.index[-1]

        # load into numpy arrays (except target var)
        x = np.array(
            [
                basin_df[variable].values
                for variable in basin_df.columns
                if variable != self.target_var
            ]
        ).T

        # y should be shape (n_times, 1)
        y = np.array(basin_df[self.target_var].values).T
        y = y.reshape(-1, 1)

        return x, y

    def _remove_nans(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Remove nans from X and y

        Args:
            x (np.ndarray): the input data to the model
            y (np.ndarray): the target data for the model

        Returns:
            Tuple[np.ndarray, np.ndarray]: x, y
        """
        # any nans in y? (2D)
        y_nans = np.any(np.isnan(y), axis=1)
        # any nans in x? (3D)
        x_nans = np.any(np.any(np.isnan(x), axis=1), axis=1)

        all_nans = np.any([y_nans, x_nans], axis=0)
        self.missing_timesteps = all_nans
        total_nans = all_nans.sum()

        y = y[~all_nans]
        x = x[~all_nans]

        if total_nans > 0:
            print(f"{total_nans} NaNs removed in Basin: {self.basin}")

        return x, y

    def _load_dynamic_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self._get_dynamic_data()

        # normalize X data
        x = (x - self.normalization_dict["dynamic_means"]) / self.normalization_dict[
            "dynamic_stds"
        ]

        # create sequences of X -> y pairs
        x, y = reshape_data(x, y, self.seq_length)

        # delete nans
        x, y = self._remove_nans(x, y)

        if self.is_train:
            # store std of discharge BEFORE normalisation
            self.q_std = np.std(y)
            y = (y - self.normalization_dict["target_mean"]) / self.normalization_dict[
                "target_std"
            ]

        # convert to torch tensors
        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32))

        return x, y

    def _get_ts_of_interest(self, basin_df: pd.DataFrame) -> pd.DataFrame:
        """Create a new dataframe with only the datetimes that will be
        included in the X data.

        Args:
            basin_df (pd.DataFrame): dynamic data

        Returns:
            pd.DataFrame: return the data only for the timeseries of interest
        """
        start_date = pd.to_datetime(f"{self.train_dates[0]}-01-01") - pd.DateOffset(
            days=self.seq_length - 1
        )
        end_date = pd.to_datetime(f"{self.train_dates[-1]}-12-31")
        basin_df = basin_df[start_date:end_date]

        return basin_df

    def _create_static_data_xr(self) -> xr.Dataset:
        #  create the static data once
        attrs_csvs = [d for d in self.attributes_dir.glob("*.csv")]
        static_dfs = [pd.read_csv(d) for d in attrs_csvs]
        # join into one dataframe
        static_df = pd.concat(static_dfs, axis=1)

        # create xr object
        static_vars = [c for c in static_df.columns if c != "gauge_id"]
        dims = ["station_id"]
        coords = {"station_id": static_df.gauge_id.iloc[:, 0].values}

        static_ds = xr.Dataset(
            {
                variable_name: (dims, static_df[variable_name].values)
                for variable_name in static_vars
            },
            coords=coords,
        )

        # Fix the coordinates
        static_ds["station_id"] = static_ds["station_id"].astype(np.dtype("int64"))
        static_ds = static_ds.sortby("station_id")

        static_ds.to_netcdf(self.data_dir / "interim/static/data.nc")

        return static_ds

    def _load_static_data(self) -> torch.Tensor:
        """Load the static attributes (catchment attributes)
        as a

        Returns:
            torch.Tensor: Static data for this basin
        """
        static_df = self._get_static_data()

        # store attributes as PyTorch Tensor
        static_data = static_df.loc[static_df.index == self.basin].values

        return torch.from_numpy(static_data.astype(np.float32))

    def _get_static_data(self) -> pd.DataFrame:
        """Return the static data as an pd.DataFrame, read from the
        xr.Dataset (netcdf file).

        Returns:
            pd.DataFrame: Static catchment attributes (static over time)
        """
        # load from the xarray object
        if (self.data_dir / "interim/static/data.nc").exists():
            static_ds = xr.open_dataset(self.data_dir / "interim/static/data.nc")
        else:
            static_ds = self._create_static_data_xr()

        # drop vars from all studies
        if self.ignore_static_vars is None:
            static_ds = static_ds.drop(
                [v for v in INVALID_ATTR if v in static_ds.data_vars]
            )
        else:
            static_ds = static_ds.drop(
                [v for v in self.ignore_static_vars if v in static_ds.data_vars]
            )

        # keep only these vars
        if self.static_variables is not None:
            static_ds = static_ds[self.static_variables]

        static_df = static_ds.to_dataframe()

        # normalise the data
        static_df = (
            static_df - self.normalization_dict["static_means"]
        ) / self.normalization_dict["static_stds"]

        return static_df


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
        with_static: bool = False,
        concat_static: bool = False,
    ):
        self.data_dir = data_dir
        self.out_file = self.data_dir / "features/features.h5"

        if not self.out_file.parents[0].exists():
            self.out_file.parents[0].mkdir(exist_ok=True, parents=True)

        if self.out_file.is_file():
            raise FileExistsError(f"File already exists at {self.out_file}")

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

        # normalisation scaler
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

            q_stds = out_f.create_dataset(
                "q_stds",
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
                total_samples = input_data.shape[0] + num_samples

                # store input / output samples
                input_data.resize((total_samples, self.seq_length, self.n_variables))
                target_data.resize((total_samples, 1))
                # (append to already existing in h5 files)
                input_data[-num_samples:, :, :] = dataset.x
                target_data[-num_samples:, :] = dataset.y

                # store std of discharge of this basin for each sample
                q_stds.resize((total_samples, 1))
                q_std_array = np.array(
                    [dataset.q_std] * num_samples, dtype=np.float32
                ).reshape(-1, 1)
                q_stds[-num_samples:, :] = q_std_array

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
            (self.x, self.y, self.sample_2_basin, self.q_stds) = self._preload_data()

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
            q_std = self.q_stds[idx]

        else:
            with h5py.File(self.h5_file, "r") as f:
                x = f["input_data"][idx]
                y = f["target_data"][idx]
                basin = f["sample_2_basin"][idx]
                basin = basin.decode("ascii")
                q_std = f["q_stds"][idx]

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
        q_std = torch.from_numpy(q_std)

        if self.with_static:
            if self.concat_static:
                return x, y, q_std
            else:
                return x, attributes, y, q_std
        else:
            return x, y, q_std

    def _preload_data(
        self
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with h5py.File(self.h5_file, "r") as f:
            x = f["input_data"][:]
            y = f["target_data"][:]
            str_arr = f["sample_2_basin"][:]
            str_arr = [x.decode("ascii") for x in str_arr]
            q_stds = f["q_stds"][:]

        return x, y, str_arr, q_stds

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


def load_static_data(
    data_dir: Path,
    static_variables: Optional[List[str]] = None,
    basins: Optional[List[str]] = None,
    drop_lat_lon: bool = True,
) -> pd.DataFrame:
    """Return the static data as an pd.DataFrame, read from the
    xr.Dataset (netcdf file).

    Returns:
        pd.DataFrame: Static catchment attributes (static over time)
    """
    # load from the xarray object
    if (data_dir / "interim/static/data.nc").exists():
        static_ds = xr.open_dataset(data_dir / "interim/static/data.nc")
    else:
        assert False, "Have you run the preprocessor? `CAMELSGBPreprocessor()`"

    # keep only these vars
    if static_variables is not None:
        static_ds = static_ds[static_variables]

    static_df = static_ds.to_dataframe()

    # drop rows of basins not contained in data set
    drop_basins = [b for b in static_df.index if b not in basins]
    static_df = static_df.drop(drop_basins, axis=0)

    # drop lat/lon col
    if drop_lat_lon:
        if all(np.isin(["gauge_lat", "gauge_lon"], static_df.columns)):
            static_df = static_df.drop(["gauge_lat", "gauge_lon"], axis=1)

    return static_df
