from src.preprocess.camels_kratzert import get_basins
from src.preprocess import CAMELSGBPreprocessor
from tests.utils import _copy_runoff_data_to_tmp_path
import numpy as np

from src.models.kratzert.main import train as train_model
from src.models.kratzert.main import evaluate as evaluate_model
import pickle
import pytest


class TestTrainModel:
    def _initialise_data(self, tmp_path):
        _copy_runoff_data_to_tmp_path(tmp_path)
        processsor = CAMELSGBPreprocessor(tmp_path, open_shapefile=False)
        processsor.preprocess()

    @pytest.mark.parametrize("with_static,concat_static", [(True, False), (True, True)])
    def test_(self, tmp_path, capsys, with_static, concat_static):
        self._initialise_data(tmp_path)

        # SETTINGS
        with_basin_str = True
        train_dates = [1970, 1975]  # chosen because there is missing data
        target_var = "discharge_spec"
        x_variables = ["precipitation", "peti"]
        static_variables = ["pet_mean", "aridity", "p_seasonality"]
        seq_length = 10
        with_static = with_static
        concat_static = concat_static
        basins = get_basins(tmp_path)
        dropout = 0.4
        hidden_size = 256
        seed = 10101
        cache = True
        use_mse = True
        batch_size = 50
        num_workers = 1
        initial_forget_gate_bias = 5
        learning_rate = 1e-3
        epochs = 1

        model = train_model(
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
            dropout=dropout,
            hidden_size=hidden_size,
            seed=seed,
            cache=cache,
            use_mse=use_mse,
            batch_size=batch_size,
            num_workers=num_workers,
            initial_forget_gate_bias=initial_forget_gate_bias,
            learning_rate=learning_rate,
            epochs=epochs,
        )

        input_size_dyn = model.input_size_dyn
        input_size_stat = model.input_size_stat
        model_path = model.model_path

        evaluate_model(
            data_dir=tmp_path,
            model_path=model_path,
            input_size_dyn=input_size_dyn,
            input_size_stat=input_size_stat,
            val_dates=train_dates,
            with_static=with_static,
            static_variables=static_variables,
            dropout=dropout,
            concat_static=concat_static,
            hidden_size=hidden_size,
            target_var=target_var,
            x_variables=x_variables,
            seq_length=seq_length,
        )

        # is the data directory correctly formatted?
        dirs = ["features", "models", "interim", "raw"]
        assert all(np.isin(dirs, [d.name for d in tmp_path.iterdir()]))

        # are the models / predictions saved properly?
        results_pkl = [f for f in (tmp_path / "models").glob("*.pkl")][0]

        if (with_static) & (not concat_static):
            assert "ealstm_results.pkl" in results_pkl.name
            assert "ealstm" in [f.name for f in (tmp_path / "models").glob("*.pt")][0]

        elif (with_static) & (concat_static):
            assert "lstm_results.pkl" in results_pkl.name
            assert "lstm" in [f.name for f in (tmp_path / "models").glob("*.pt")][0]
            assert (
                "ealstm" not in [f.name for f in (tmp_path / "models").glob("*.pt")][0]
            )

        if train_dates == [1970, 1975]:
            # check that 96004 has no data and this has been captured in the
            # stdout
            captured = capsys.readouterr()
            expected_stdout = f"No data for basin 96004 in val_period: [{train_dates[0]} {train_dates[-1]}]"
            assert (
                expected_stdout in captured.out
            ), f"Expected stdout to be {expected_stdout}, got {captured.out}"

        # check that all basins are found as keys in results Dict
        results = pickle.load(open(results_pkl, "rb"))
        if train_dates == [1970, 1975]:
            assert basins[-1] in [k for k in results.keys()]
        else:
            # otherwise all basins should be in !
            assert all(np.isin(basins, [k for k in results.keys()]))
