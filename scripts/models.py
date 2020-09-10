from typing import List, Tuple
import sys

sys.path.append("..")

from src.models import (
    Persistence,
    LinearRegression,
    LinearNetwork,
    RecurrentNetwork,
    EARecurrentNetwork,
    load_model,
)
from src.analysis import all_explanations_for_file

from scripts.utils import get_data_path
import xarray as xr


def get_forecast_vars() -> List[str]:
    forecast_vars = []
    leadtimes = [1, 2, 3]
    variables = ["t2m", "tprate", "erate"]
    for variable in variables:
        for leadtime in leadtimes:
            forecast_vars.append(f"{variable}_std_{leadtime}")

    return forecast_vars


def get_ignore_static_vars() -> List[str]:
    return [
        "lc_class",  # remove for good clustering (?)
        "lc_class_group",  # remove for good clustering (?)
        "slt",  #  remove for good clustering (?)
        "no_data_one_hot",
        "lichens_and_mosses_one_hot",
        "permanent_snow_and_ice_one_hot",
        "urban_areas_one_hot",
        "water_bodies_one_hot",
    ]


def get_all_vars(experiment: str = "one_month_forecast") -> Tuple[List[str], str]:
    data_dir = get_data_path()
    # get x vars from features dir
    x_data_path = [d for d in (data_dir / f"features/{experiment}/test").glob("*/x.nc")][0]
    y_data_path = [d for d in (data_dir / f"features/{experiment}/test").glob("*/y.nc")][0]
    x_ds = xr.open_dataset(x_data_path)
    y_ds = xr.open_dataset(y_data_path)
    return [v for v in x_ds.data_vars], [v for v in y_ds.data_vars][0]


def persistence(experiment="one_month_forecast",):
    data_path = get_data_path()
    spatial_mask = data_path / "interim/boundaries_preprocessed/kenya_asal_mask.nc"
    spatial_mask = None
    predictor = Persistence(data_path, experiment=experiment, spatial_mask=spatial_mask)
    predictor.evaluate(save_preds=True)


def regression(
    experiment="one_month_forecast",
    include_pred_month=True,
    surrounding_pixels=None,
    ignore_vars=None,
):
    data_path = get_data_path()
    spatial_mask = data_path / "interim/boundaries_preprocessed/kenya_asal_mask.nc"
    spatial_mask = None

    predictor = LinearRegression(
        data_path,
        experiment=experiment,
        include_pred_month=include_pred_month,
        surrounding_pixels=surrounding_pixels,
        ignore_vars=ignore_vars,
        static="embeddings",
        spatial_mask=spatial_mask,
    )
    predictor.train()
    predictor.evaluate(save_preds=True)

    # mostly to test it works
    # predictor.explain(save_shap_values=True)


def linear_nn(
    experiment="one_month_forecast",
    include_pred_month=True,
    surrounding_pixels=None,
    ignore_vars=None,
    pretrained=False,
    static=None,
):
    predictor = LinearNetwork(
        layer_sizes=[100],
        data_folder=get_data_path(),
        experiment=experiment,
        include_pred_month=include_pred_month,
        surrounding_pixels=surrounding_pixels,
        ignore_vars=ignore_vars,
        static=static,
    )
    predictor.train(num_epochs=50, early_stopping=5)
    predictor.evaluate(save_preds=True)
    predictor.save_model()

    # _ = predictor.explain(save_shap_values=True)


def rnn(
    experiment="one_month_forecast",
    include_pred_month=True,
    surrounding_pixels=None,
    ignore_vars=None,
    pretrained=False,
    static=None,
):
    predictor = RecurrentNetwork(
        hidden_size=128,
        data_folder=get_data_path(),
        experiment=experiment,
        include_pred_month=include_pred_month,
        surrounding_pixels=surrounding_pixels,
        ignore_vars=ignore_vars,
        static=static,
    )
    predictor.train(num_epochs=50, early_stopping=5)
    predictor.evaluate(save_preds=True)
    predictor.save_model()

    # _ = predictor.explain(save_shap_values=True)


def earnn(
    experiment="one_month_forecast",
    include_pred_month=True,
    surrounding_pixels=None,
    pretrained=False,
    ignore_vars=None,
    static="embeddings",
    explain: bool = False,
):
    data_path = get_data_path()

    if static is None:
        print("** Cannot fit EALSTM without spatial information **")
        return

    if not pretrained:
        predictor = EARecurrentNetwork(
            hidden_size=128,
            data_folder=data_path,
            experiment=experiment,
            include_pred_month=include_pred_month,
            surrounding_pixels=surrounding_pixels,
            ignore_vars=ignore_vars,
            static=static,
        )
        predictor.train(num_epochs=50, early_stopping=5)
        predictor.evaluate(save_preds=True)
        predictor.save_model()
    else:
        predictor = load_model(data_path / f"models/{experiment}/ealstm/model.pt")

    test_file = data_path / f"features/{experiment}/test/2018_3"
    assert test_file.exists()

    if explain:
        all_explanations_for_file(test_file, predictor, batch_size=100)


if __name__ == "__main__":
    experiment = "nowcast"

    # ignore_vars = ["VCI", "p84.162", "sp", "tp", "VCI1M"]
    forecast_vars = get_forecast_vars()
    ignore_static_vars = get_ignore_static_vars()

    x_vars, y_var = get_all_vars(experiment=experiment)
    ignore_dyn_vars = [
        v for v in x_vars
        if (v not in forecast_vars) or (v != y_var)
    ]

    ignore_vars = ignore_dyn_vars + ignore_static_vars
    assert False

    persistence(experiment=experiment)
    # regression(experiment=experiment, ignore_vars=ignore_vars)
    linear_nn(experiment=experiment, ignore_vars=ignore_vars, static="features")
    rnn(experiment=experiment, ignore_vars=ignore_vars, static="features")
    earnn(experiment=experiment, ignore_vars=ignore_vars, static="features")
