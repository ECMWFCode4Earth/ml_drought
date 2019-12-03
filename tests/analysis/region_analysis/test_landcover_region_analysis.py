import numpy as np
import pandas as pd
import xarray as xr

from src.analysis import LandcoverRegionAnalysis
from tests.utils import _make_dataset
from .test_base import TestRegionAnalysis


class TestLandcoverRegionAnalysis(TestRegionAnalysis):
    @staticmethod
    def _create_dummy_landcover_data(tmp_path):
        parent_dir = tmp_path / "interim" / "static" / "esa_cci_landcover_preprocessed"
        parent_dir.mkdir(exist_ok=True, parents=True)
        fname = "esa_cci_landcover_kenya_one_hot.nc"
        vars = [
            "Cropland, irrigated or post-flooding_one_hot",
            "Herbaceous cover_one_hot",
            "No data_one_hot",
            "Tree or shrub cover_one_hot",
        ]
        # create non-overlapping groups
        # https://stackoverflow.com/a/52356978/9940782
        groups = np.random.randint(0, 4, (30, 30))
        masks = (groups[..., None] == np.arange(4)[None, :]).T.astype(int)

        all_ds = []
        for group, var in enumerate(vars):
            ds, _, _ = _make_dataset(
                (30, 30),
                variable_name=var,
                lonmin=30,
                lonmax=35,
                latmin=-2,
                latmax=2,
                add_times=False,
                const=True,
            )
            # assign the values from the mask to the da.values
            ds[var].values = masks[group, :, :]
            all_ds.append(ds)

        ds = xr.merge([*all_ds])
        ds.to_netcdf(parent_dir / fname)

    def test_loading_functions(self, tmp_path):
        self._create_dummy_true_preds_data(tmp_path)
        self._create_dummy_landcover_data(tmp_path)

        analyser = LandcoverRegionAnalysis(data_dir=tmp_path)

        region_data_dir = (
            tmp_path / "interim" / "static" / "esa_cci_landcover_preprocessed"
        )
        region_data_path = region_data_dir / "esa_cci_landcover_kenya_one_hot.nc"
        landcover_das = analyser.load_landcover_data(region_data_path)

        true_data_dir = tmp_path / "features" / "one_month_forecast" / "test"
        pred_data_dir = tmp_path / "models" / "one_month_forecast"
        true_data_paths = [f for f in true_data_dir.glob("**/y.nc")]
        pred_data_paths = [f for f in pred_data_dir.glob("**/*.nc")]
        true_data_paths.sort()
        pred_data_paths.sort()

        true_data_path = true_data_paths[0]
        pred_data_path = pred_data_paths[0]
        true_da = analyser.load_prediction_data(pred_data_path)
        pred_da = analyser.load_true_data(true_data_path)

        assert pred_da.time == true_da.time, (
            "the predicted time should be the " "same as the true data time."
        )

        assert len(landcover_das) == 4
        assert all([isinstance(da, xr.DataArray) for da in landcover_das])

    def test_compute_mean_stats(self, tmp_path):
        # create and load all data
        self._create_dummy_true_preds_data(tmp_path)
        self._create_dummy_landcover_data(tmp_path)

        region_data_dir = (
            tmp_path / "interim" / "static" / "esa_cci_landcover_preprocessed"
        )
        region_data_path = region_data_dir / "esa_cci_landcover_kenya_one_hot.nc"
        true_data_dir = tmp_path / "features" / "one_month_forecast" / "test"
        pred_data_dir = tmp_path / "models" / "one_month_forecast"
        true_data_paths = [f for f in true_data_dir.glob("**/*.nc")]
        pred_data_paths = [f for f in pred_data_dir.glob("**/*.nc")]
        true_data_paths.sort()
        pred_data_paths.sort()

        true_data_path = true_data_paths[0]
        pred_data_path = pred_data_paths[0]

        analyser = LandcoverRegionAnalysis(tmp_path)
        true_da = analyser.load_prediction_data(pred_data_path)
        pred_da = analyser.load_true_data(true_data_path)
        datetime = pd.to_datetime(pred_da.time.values).to_pydatetime()
        landcover_das = analyser.load_landcover_data(region_data_path)

        (
            datetimes,
            region_name,
            predicted_mean_value,
            true_mean_value,
        ) = analyser.compute_mean_statistics(
            landcover_das=landcover_das,
            pred_da=pred_da,
            true_da=true_da,
            datetime=datetime,
        )
        assert len(datetimes) == len(region_name) == len(predicted_mean_value)
        assert len(predicted_mean_value) == len(true_mean_value)

    def test_analyzer_analyze(self, tmp_path):
        self._create_dummy_landcover_data(tmp_path)
        self._create_dummy_true_preds_data(tmp_path)

        # admin_boundaries=False for landcover grouping
        analyser = LandcoverRegionAnalysis(tmp_path)
        assert "landcover_kenya_one_hot" in analyser.region_data_paths[0].name

        lcover_das = analyser.load_landcover_data(analyser.region_data_paths[0])
        assert isinstance(lcover_das, list)

        analyser._analyze_single(analyser.region_data_paths[0])

        csv_path = (
            tmp_path
            / "analysis"
            / "region_analysis"
            / "ealstm"
            / "ealstm_landcover.csv"
        )
        assert (csv_path).exists()

        df = pd.read_csv(csv_path)
        assert df.admin_level_name.unique() == ["landcover"]

        valid_landcover_names = [
            "cropland_irrigated_or_postflooding_one_hot",
            "herbaceous_cover_one_hot",
            "no_data_one_hot",
            "tree_or_shrub_cover_one_hot",
        ]
        assert np.isin(valid_landcover_names, df.region_name.unique()).all()

        n_datetimes = 3
        n_lc_regions = len(valid_landcover_names)
        assert len(df) == (n_datetimes * n_lc_regions * len(["landcover"]))
