import numpy as np
import pandas as pd
import pytest

from src.models.base import ModelBase


class TestBase:
    def test_init(self, tmp_path):
        ModelBase(tmp_path)
        assert (tmp_path / "models").exists(), f"Models dir not made!"

    @pytest.mark.parametrize(
        "save_preds,predict_delta,check_inverted",
        [
            (True, True, False),
            (False, True, False),
            (True, False, False),
            (False, False, False),
            (True, True, True),
            (False, True, True),
            (True, False, True),
            (False, False, True),
        ],
    )
    def test_evaluate(
        self, tmp_path, monkeypatch, capsys, save_preds, predict_delta, check_inverted
    ):
        def mockreturn(self):

            y = np.array([1, 1, 1, 1, 1])
            latlons = [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5]]
            latlons = np.array([np.array(xi) for xi in latlons])
            y_var = "VCI"
            time = pd.to_datetime("2011-01-01")
            if predict_delta:
                test_arrays = {
                    "hello": {
                        "y": y,
                        "historical_target": y,
                        "latlons": latlons,
                        "time": time,
                        "y_var": y_var,
                    }
                }
            else:
                test_arrays = {
                    "hello": {"y": y, "latlons": latlons, "time": time, "y_var": y_var}
                }
            preds_arrays = {"hello": y}

            return test_arrays, preds_arrays

        monkeypatch.setattr(ModelBase, "predict", mockreturn)

        base = ModelBase(tmp_path, predict_delta=predict_delta)
        model_dir = tmp_path / "models" / "base"
        if not model_dir.exists():
            model_dir.mkdir(exist_ok=True, parents=True)
        base.model_dir = model_dir
        base.evaluate(
            save_results=False, save_preds=save_preds, check_inverted=check_inverted
        )

        captured = capsys.readouterr()
        expected_stdout = "RMSE: 0.0"
        assert (
            expected_stdout in captured.out
        ), f"Expected stdout to be {expected_stdout}, got {captured.out}"
