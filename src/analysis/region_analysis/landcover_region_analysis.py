from pathlib import Path
import xarray as xr
from datetime import datetime
from typing import List, Tuple
import string

from .base import RegionAnalysis


class LandcoverRegionAnalysis(RegionAnalysis):
    admin_boundaries = False

    def __init__(
        self, data_dir: Path = Path("data"), experiment: str = "one_month_forecast"
    ):

        super().__init__(
            data_dir=data_dir,
            experiment=experiment,
            admin_boundaries=self.admin_boundaries,
        )

    @staticmethod
    def create_lc_name(landcover_name: str) -> str:
        rm_punctuation = str.maketrans("", "", string.punctuation)
        return (
            landcover_name.lower()
            .replace("_", " ")
            .translate(rm_punctuation)
            .replace(" ", "_")
        )

    def load_landcover_data(self, region_data_path: Path) -> List[xr.DataArray]:
        # load the one_hot_encoded preprocessed landcover data
        landcover_ds: xr.Dataset = xr.open_dataset(region_data_path)
        if "lc_class" in [d for d in landcover_ds.data_vars]:
            landcover_ds = landcover_ds.drop("lc_class")
        assert all(
            ["one_hot" in var for var in landcover_ds.data_vars]
        ), "This method only works with one_hot_encoded landcover data"

        # return a list of DataArrays (boolean masks)
        landcover_das = [landcover_ds[v] for v in landcover_ds.data_vars]

        return landcover_das

    def compute_mean_statistics(
        self,
        landcover_das: List[xr.DataArray],
        pred_da: xr.DataArray,
        true_da: xr.DataArray,
        datetime: datetime,
    ) -> Tuple[List, List, List, List]:
        """
        Returns:
        --------
        datetimes: List
            datetime of the true/predicted values
        region_names: List
            the name of the region
        predicted_mean_value: List
            the mean predicted value for ROI
        true_mean_value: List
            the mean true value for ROI
        """
        # For each region calculate mean `target_variable` in true / pred
        region_names: List = []
        predicted_mean_value: List = []
        true_mean_value: List = []
        datetimes: List = []

        for landcover_da in landcover_das:
            lc_name = self.create_lc_name(landcover_da.name)
            region_names.append(lc_name)
            # because one-hot-encoded only select where value == 1
            predicted_mean_value.append(pred_da.where(landcover_da == 1).mean().values)
            true_mean_value.append(true_da.where(landcover_da == 1).mean().values)
            # assert true_da.time == pred_da.time, 'time must be matching!'
            datetimes.append(datetime)

        assert len(region_names) == len(predicted_mean_value) == len(datetimes)
        return datetimes, region_names, predicted_mean_value, true_mean_value

    def _analyze_single(self, region_data_path: Path):
        landcover_das = self.load_landcover_data(region_data_path)

        admin_level_name = "landcover"
        return self._base_analyze_single(
            admin_level_name=admin_level_name, landcover_das=landcover_das
        )

    def analyze(
        self,
        compute_global_errors: bool = True,
        compute_regional_errors: bool = True,
        save_all_df: bool = True,
    ) -> None:
        """For all preprocessed regions calculate the mean True value and
        mean predicted values. Also have the option to calculate global
        errors (across all regions (in an admin level) / landcover classes).

           admin_level_name   model   datetime region_name      pred... true...
        0         landcover  ealstm 2018-01-31  lctype_1        48.16   50.67
        1         landcover  ealstm 2018-01-31  lctype_2        50.96   49.66
        2         landcover  ealstm 2018-01-31  lctype_3        47.65   47.61
        3         landcover  ealstm 2018-01-31  lctype_4        50.79   47.00
        4         landcover  ealstm 2018-02-28  lctype_1        50.78   53.47
        """
        # should only be one landcover file
        assert len(self.region_data_paths) == 1
        region_data_path = self.region_data_paths[0]
        df = self._analyze_single(region_data_path)

        if "index" in df.columns:
            df = df.drop(columns="index")
        if "level_0" in df.columns:
            df = df.drop(columns="level_0")

        self.df = df.astype(
            {"predicted_mean_value": "float64", "true_mean_value": "float64"}
        )
        print("* Assigned all region dfs to `self.df` *")

        # compute error metrics for each model globally
        self.compute_metrics(compute_global_errors, compute_regional_errors)

        if save_all_df:
            self.df.to_csv(self.out_dir / f"{self.experiment}_all_admin_regions.csv")
