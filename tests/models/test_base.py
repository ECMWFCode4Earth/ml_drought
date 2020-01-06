import numpy as np
import pytest

from src.models.base import ModelBase


class TestBase:
    def test_init(self, tmp_path):
        ModelBase(tmp_path)
        assert (tmp_path / "models").exists(), f"Models dir not made!"

    @pytest.mark.parametrize(
        "save_preds,predict_delta",
        [(True, True), (False, True), (True, False), (False, False)]
    )
    def test_evaluate(self, tmp_path, monkeypatch, capsys, save_preds, predict_delta):
        def mockreturn(self):

            y = np.array([1, 1, 1, 1, 1])

            test_arrays = {"hello": {"y": y}}
            preds_arrays = {"hello": y}

            return test_arrays, preds_arrays

        monkeypatch.setattr(ModelBase, "predict", mockreturn)

        base = ModelBase(tmp_path, predict_delta=predict_delta)
        base.evaluate(save_results=False, save_preds=save_preds)

        captured = capsys.readouterr()
        expected_stdout = "RMSE: 0.0"
        assert (
            expected_stdout in captured.out
        ), f"Expected stdout to be {expected_stdout}, got {captured.out}"
