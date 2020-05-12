import numpy as np
import pandas as pd

from src.analysis import AdministrativeRegionAnalysis
from tests.utils import _make_dataset
from .test_base import TestRegionAnalysis


class TestAdministrativeRegionAnalysis(TestRegionAnalysis):
    @staticmethod
    def _create_dummy_admin_boundaries_data(tmp_path, prefix: str):
        ds, _, _ = _make_dataset(
            (30, 30),
            variable_name="VHI",
            lonmin=30,
            lonmax=35,
            latmin=-2,
            latmax=2,
            add_times=False,
        )
        ds.VHI.astype(int)

        (tmp_path / "analysis" / "boundaries_preprocessed").mkdir(
            exist_ok=True, parents=True
        )
        ds.attrs["keys"] = ", ".join([str(i) for i in range(3)])
        ds.attrs["values"] = ", ".join([f"region_{i}" for i in np.arange(0, 3)])
        ds.to_netcdf(
            tmp_path
            / "analysis"
            / "boundaries_preprocessed"
            / f"province_l{prefix}_kenya.nc"
        )

    def test_loading_functions(self, tmp_path):
        self._create_dummy_true_preds_data(tmp_path)
        self._create_dummy_admin_boundaries_data(tmp_path, prefix="1")

        analyser = AdministrativeRegionAnalysis(data_dir=tmp_path)

        region_data_dir = tmp_path / "analysis" / "boundaries_preprocessed"
        region_data_path = region_data_dir / f"province_l1_kenya.nc"
        region_da, region_dict, region_name = analyser.load_region_data(
            region_data_path
        )

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

        assert region_name == "province_l1_kenya.nc"
        assert [k for k in region_dict.keys()] == [0, 1, 2]
        assert [v for v in region_dict.values()] == ["region_0", "region_1", "region_2"]

    def test_compute_mean_stats(self, tmp_path):
        # create and load all data
        self._create_dummy_true_preds_data(tmp_path)
        self._create_dummy_admin_boundaries_data(tmp_path, prefix="1")

        region_data_dir = tmp_path / "analysis" / "boundaries_preprocessed"
        region_data_path = region_data_dir / f"province_l1_kenya.nc"
        true_data_dir = tmp_path / "features" / "one_month_forecast" / "test"
        pred_data_dir = tmp_path / "models" / "one_month_forecast"
        true_data_paths = [f for f in true_data_dir.glob("**/*.nc")]
        pred_data_paths = [f for f in pred_data_dir.glob("**/*.nc")]
        true_data_paths.sort()
        pred_data_paths.sort()

        true_data_path = true_data_paths[0]
        pred_data_path = pred_data_paths[0]

        analyser = AdministrativeRegionAnalysis(tmp_path)
        true_da = analyser.load_prediction_data(pred_data_path)
        pred_da = analyser.load_true_data(true_data_path)
        datetime = pd.to_datetime(pred_da.time.values).to_pydatetime()
        region_da, region_lookup, region_name = analyser.load_region_data(
            region_data_path
        )

        (
            datetimes,
            region_name,
            predicted_mean_value,
            true_mean_value,
        ) = analyser.compute_mean_statistics(
            region_da=region_da,
            region_lookup=region_lookup,
            pred_da=pred_da,
            true_da=true_da,
            datetime=datetime,
        )
        assert len(datetimes) == len(region_name) == len(predicted_mean_value)
        assert len(predicted_mean_value) == len(true_mean_value)

    def test_analyze_single(self, tmp_path):
        # NOTE: all inputs (region_da, preds_da, true_da) need the same shape
        #  for the analyzer to work properly
        self._create_dummy_true_preds_data(tmp_path)
        self._create_dummy_admin_boundaries_data(tmp_path, prefix="1")

        region_data_dir = tmp_path / "analysis" / "boundaries_preprocessed"
        region_data_path = region_data_dir / f"province_l1_kenya.nc"

        analyser = AdministrativeRegionAnalysis(tmp_path)
        analyser._analyze_single(region_data_path=region_data_path)

        model = "ealstm"
        admin_level_name = "province_l1_kenya"
        out_dir = tmp_path / "analysis/region_analysis"
        df = pd.read_csv(out_dir / model / f"{model}_{admin_level_name}.csv")

        # check the dataframe outputted
        assert df.iloc[0].datetime == "2018-01-31"
        assert np.isin(
            ["region_0", "region_1", "region_2"], df.region_name.unique()
        ).all()
        assert np.isin(
            ["datetime", "region_name", "predicted_mean_value", "true_mean_value"],
            df.columns,
        ).all()

    def test_analyzer_analyze(self, tmp_path):
        # create the dummy data
        for i in range(3):
            self._create_dummy_admin_boundaries_data(tmp_path, prefix=str(i))
        self._create_dummy_true_preds_data(tmp_path)

        # test the analyzer
        analyser = AdministrativeRegionAnalysis(tmp_path)
        analyser.analyze()

        output_paths = [
            f for f in (tmp_path / "analysis/region_analysis/ealstm").iterdir()
        ]
        assert len(output_paths) == 3

        for output_path in output_paths:
            # check for EVERY region
            df = pd.read_csv(output_path)
            assert np.isin(
                ["region_0", "region_1", "region_2"], df.region_name.unique()
            ).all()
            assert np.isin(
                ["datetime", "region_name", "predicted_mean_value", "true_mean_value"],
                df.columns,
            ).all()

        # check the main df saved as a class attribute
        assert analyser.df.model.unique() == ["ealstm"]
        assert isinstance(analyser.df, pd.DataFrame)

        # the number of rows in the dataframe
        num_models = 1
        num_datetimes = num_regions = num_admin_levels = 3
        expected_number_rows = (
            num_models * num_admin_levels * num_datetimes * num_regions
        )
        assert len(analyser.df) == expected_number_rows, (
            f"Expected {expected_number_rows}"
            f"Got: {len(analyser.df)}. We should have a row for each combination"
            "of models, datetimes, regions, admin_levels"
        )
