from src.engineer.basin import CAMELSCSV
from src.engineer.runoff import RunoffEngineer
from ..utils import _copy_runoff_data_to_tmp_path
from src.preprocess import CAMELSGBPreprocessor
from src.engineer.runoff_utils import CalculateNormalizationParams


class TestCAMELSCSV:
    def test(self, tmp_path):
        # preprocess the data
        _copy_runoff_data_to_tmp_path(tmp_path)
        processsor = CAMELSGBPreprocessor(tmp_path, open_shapefile=False)
        processsor.preprocess()

        # settings for experiment
        train_dates = [2000, 2001]
        target_var = "discharge_spec"
        x_variables = ["precipitation", "peti"]
        static_variables = ["pet_mean", "aridity", "p_seasonality"]
        seq_length = 10
        with_static = True
        is_train = True
        concat_static = False

        # Calculate normalization
        normalization_dict = CalculateNormalizationParams(
            data_dir=tmp_path,
            train_dates=train_dates,
            target_var=target_var,
            x_variables=x_variables,
            static_variables=static_variables,
        ).normalization_dict

        # Test the individual basin X, y engineer
        for basin in get_basins(tmp_path):
            dataset = CAMELSCSV(
                data_dir=tmp_path,
                basin=basin,
                train_dates=train_dates,
                normalization_dict=normalization_dict,
                is_train=is_train,
                target_var=target_var,
                x_variables=x_variables,
                static_variables=static_variables,
                seq_length=seq_length,
                with_static=with_static,
                concat_static=concat_static,
            )
            x = dataset.x
            y = dataset.y
            static = dataset.attributes
            scaler = dataset.normalization_dict

            assert x.shape == (n_times, seq_length, n_features)
            assert y.shape == (n_times, 1)
            assert static.shape == (1, n_static_features)

        # TEST NORMALIZATION DICT
        expected = [
            "static_means",
            "static_stds",
            "target_mean",
            "target_std",
            "dynamic_stds",
            "dynamic_means",
            "x_variables",
            "target_var",
            "static_variables",
        ]
        assert all(
            np.isin([k for k in scaler.keys()], expected)
        ), f"Expected: {expected} Got: {[k for k in scaler.keys()]}"

        # DERIVED Values
        n_times = len(
            pd.date_range(
                f"{train_dates[0]}-01-01", f"{train_dates[-1]}-12-31", freq="D"
            )
        )
        n_features = len(x_variables)
        n_stations = 2
        n_static_features = len(static_variables)

        assert len([stn for stn in get_basins(tmp_path)]) == n_stations


class TestRunoffEngineer:
    def test(self, tmp_path):
        # preprocess the data
        _copy_runoff_data_to_tmp_path(tmp_path)
        processsor = CAMELSGBPreprocessor(tmp_path, open_shapefile=False)
        processsor.preprocess()

        # SETTINGS
        train_dates = [2000, 2001]
        target_var = "discharge_spec"
        x_variables = ["precipitation", "peti"]
        static_variables = ["pet_mean", "aridity", "p_seasonality"]
        seq_length = 10
        with_static = True
        concat_static = False
        basins = get_basins(tmp_path)
        with_basin_str = True

        # INITIALIZE engineer
        engineer = RunoffEngineer(
            data_dir=tmp_path,
            basins=basins,
            train_dates=train_dates,
            with_basin_str=with_basin_str,
            target_var=target_var,
            x_variables=x_variables,
            static_variables=static_variables,
            ignore_static_vars=None,
            seq_length=seq_length,
            with_static=with_static,
            concat_static=concat_static,
        )

        engineer.create_training_data()
        h5_file = engineer.out_file

        assert h5_file.exists()
        with h5py.File(h5_file, "r") as f:
            x = f["input_data"][:]
            y = f["target_data"][:]
            str_arr = f["sample_2_basin"][:]
            str_arr = [x.decode("ascii") for x in str_arr]
            q_stds = f["q_stds"][:]

        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(str_arr, list)
        assert isinstance(q_stds, np.ndarray)

        assert len(np.unique(q_stds)) == 2
        assert len(np.unique(str_arr)) == 2
        assert x[0].shape == (seq_length, len(x_variables))
        assert len(x) == len(y)
