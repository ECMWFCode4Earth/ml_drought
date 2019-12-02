import pickle

from src.analysis.plot_explanations import npy_to_netcdf, all_explanations_for_file
from src.models import LinearNetwork

from tests.utils import _make_dataset


class TestPlotExplanations:

    def test_npy_to_netcdf(self, tmp_path):

        x, _, _ = _make_dataset(size=(5, 5), const=True)
        y = x.isel(time=[-1])

        train_features = tmp_path / "features/one_month_forecast/train/hello"
        train_features.mkdir(parents=True)

        test_features = tmp_path / "features/one_month_forecast/test/hello"
        test_features.mkdir(parents=True)

        norm_dict = {"VHI": {"mean": 0.0, "std": 1.0}}
        with (tmp_path / "features/one_month_forecast/normalizing_dict.pkl").open(
            "wb"
        ) as f:
            pickle.dump(norm_dict, f)

        x.to_netcdf(test_features / "x.nc")
        y.to_netcdf(test_features / "y.nc")

        x.to_netcdf(train_features / "x.nc")
        y.to_netcdf(train_features / "y.nc")

        # static
        x_static, _, _ = _make_dataset(size=(5, 5), add_times=False)
        static_features = tmp_path / f"features/static"
        static_features.mkdir(parents=True)
        x_static.to_netcdf(static_features / "data.nc")

        static_norm_dict = {"VHI": {"mean": 0.0, "std": 1.0}}
        with (tmp_path / f"features/static/normalizing_dict.pkl").open("wb") as f:
            pickle.dump(static_norm_dict, f)

        model = LinearNetwork(layer_sizes=[5], data_folder=tmp_path)
        model.train()
        # all_explanations_for_file(tmp_path / "features/one_month_forecast/test/hello", model)

        test_dl = next(
            iter(model.get_dataloader(mode="test", to_tensor=False, shuffle_data=False))
        )

        for _, dl in test_dl.items():
            _ = npy_to_netcdf(dl, dl.x.historical)
