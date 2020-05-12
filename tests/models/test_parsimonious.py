from src.models.persistence import Persistence

from ..utils import _make_dataset


class TestPersistence:
    def test_predict(self, tmp_path):

        x, _, _ = _make_dataset(size=(5, 5))
        y = x.isel(time=[-1])

        test_features = tmp_path / "features/one_month_forecast/test/1980_1"
        test_features.mkdir(parents=True)

        x.to_netcdf(test_features / "x.nc")
        y.to_netcdf(test_features / "y.nc")

        predictor = Persistence(tmp_path)

        test_arrays, preds = predictor.predict()

        assert (
            test_arrays["1980_1"]["y"] == preds["1980_1"]
        ).all(), f"Last timestep not correctly taken!"
