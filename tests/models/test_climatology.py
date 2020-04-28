from src.models.climatology import Climatology

from ..utils import _make_dataset


class TestClimatology:
    @staticmethod
    def _create_train_samples(tmp_path, n_samples: int = 12):
        for i in range(n_samples % 13):
            # create datasets
            x, _, _ = _make_dataset(size=(5, 5))
            y = x.isel(time=[-1])
            # write to nc
            test_features = tmp_path / f"features/one_month_forecast/train/2000_{i}"
            test_features.mkdir(parents=True)

            x.to_netcdf(test_features / "x.nc")
            y.to_netcdf(test_features / "y.nc")

    def test_predict(self, tmp_path):
        self._create_train_samples(tmp_path, 12)

        x, _, _ = _make_dataset(size=(5, 5))
        y = x.isel(time=[-1])

        test_features = tmp_path / "features/one_month_forecast/test/1980_1"
        test_features.mkdir(parents=True)

        x.to_netcdf(test_features / "x.nc")
        y.to_netcdf(test_features / "y.nc")

        predictor = Climatology(tmp_path)

        test_arrays, preds = predictor.predict()

        assert (
            test_arrays["1980_1"]["y"] == preds["1980_1"]
        ).all(), f"Last timestep not correctly taken!"
