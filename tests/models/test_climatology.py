from src.models.climatology import Climatology
from src.analysis import read_train_data
from ..utils import _make_dataset


class TestClimatology:
    @staticmethod
    def _create_train_samples(tmp_path, n_samples: int = 12):
        for i in range(n_samples % 13):
            # create datasets
            x, _, _ = _make_dataset(size=(5, 5), end_date=f"2000-{i+1}")
            y = x.isel(time=[-1])
            # write to nc
            test_features = tmp_path / f"features/one_month_forecast/train/2000_{i+1}"
            test_features.mkdir(parents=True)

            x.to_netcdf(test_features / "x.nc")
            y.to_netcdf(test_features / "y.nc")

    def test_predict(self, tmp_path):
        self._create_train_samples(tmp_path, 12)

        # TEST with a random nan value included!
        x, _, _ = _make_dataset(size=(5, 5), random_nan=1)
        y = x.isel(time=[-1])

        test_features = tmp_path / "features/one_month_forecast/test/1980_1"
        test_features.mkdir(parents=True)

        x.to_netcdf(test_features / "x.nc")
        y.to_netcdf(test_features / "y.nc")

        predictor = Climatology(tmp_path)

        test_arrays, preds = predictor.predict()

        assert (
            test_arrays["1980_1"]["y"].shape == preds["1980_1"].shape
        ), f"Shape of climatology is incorrect!"

        # calculate climatology
        _, y_train = read_train_data(tmp_path)
        ds = y_train
        nan_mask = test_arrays["1980_1"]["nan_mask"]

        # check that the nan mask is 1 (the random nan value we included!)
        assert nan_mask.sum() == 1

        assert (
            preds["1980_1"].flatten()
            == ds["VHI"]["time.month" == 1].values.flatten()[~nan_mask]
        ).all(), "Expect the month mean to be the calculated from the training data"
