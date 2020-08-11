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
    pretrained=True,
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
    pretrained=True,
    ignore_vars=None,
    static="embeddings",
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
    all_explanations_for_file(test_file, predictor, batch_size=100)


if __name__ == "__main__":
    # ignore_vars = ["VCI", "p84.162", "sp", "tp", "VCI1M"]
    ignore_vars = None
    persistence()
    # regression(ignore_vars=ignore_vars)
    # linear_nn(ignore_vars=ignore_vars, static=None)
    rnn(ignore_vars=ignore_vars, static=None)
    earnn(ignore_vars=ignore_vars, static=None)
