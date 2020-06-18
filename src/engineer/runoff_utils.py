from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
from typing import Dict, List, Tuple, Optional, DefaultDict
import pickle
from collections import defaultdict


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

        self.output_folder = data_dir / "features/"
        self.output_folder.mkdir(exist_ok=True, parents=True)

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
        static_ds = static_ds[static_variables]

        self.normalization_dict = self.calculate_normalization_dict(
            dynamic_ds=dynamic_ds,
            static_ds=static_ds,
            train_dates=self.train_dates,
            x_variables=self.x_variables,
            static_variables=self.static_variables,
            target_var=self.target_var,
        )

        self.static_normalization_values = self.calculate_var_normalization_values(
            ds=static_ds, reducing_dims=["station_id"]
        )
        pickle.dump(
            self.static_normalization_values,
            open(self.output_folder / "static_normalization_values.pkl", "wb"),
        )

        self.dynamic_normalization_values = self.calculate_var_normalization_values(
            ds=dynamic_ds, reducing_dims=["time", "station_id"]
        )
        pickle.dump(
            self.dynamic_normalization_values,
            open(self.output_folder / "dynamic_normalization_values.pkl", "wb"),
        )

    @staticmethod
    def calculate_var_normalization_values(
        ds: xr.Dataset, reducing_dims: Optional[List[str]] = ["time", "station_id"]
    ) -> DefaultDict[str, Dict[str, float]]:
        normalization_values: DefaultDict[str, Dict[str, float]] = defaultdict(dict)

        for var in ds.data_vars:
            if var.endswith("one_hot"):
                mean = 0.0
                std = 1.0
            else:
                mean = float(ds[var].mean(dim=reducing_dims, skipna=True).values)
                std = float(ds[var].std(dim=reducing_dims, skipna=True).values)

            normalization_values[var]["mean"] = mean
            normalization_values[var]["std"] = std

        return normalization_values

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
