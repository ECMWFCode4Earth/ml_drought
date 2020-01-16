from pathlib import Path
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


def parsimonious(experiment="one_month_forecast",):

    data_path = get_data_path()
    predictor = Persistence(
        data_path,
        experiment=experiment,
        spatial_mask=data_path / "interim/boundaries_preprocessed/kenya_asal_mask.nc",
    )
    predictor.evaluate(save_preds=True)


def regression(
    experiment="one_month_forecast", include_pred_month=True, surrounding_pixels=None
):

    data_path = get_data_path()
    predictor = LinearRegression(
        data_path,
        experiment=experiment,
        include_pred_month=include_pred_month,
        surrounding_pixels=surrounding_pixels,
        static="embeddings",
        spatial_mask=data_path / "interim/boundaries_preprocessed/kenya_asal_mask.nc",
    )
    predictor.train()
    predictor.evaluate(save_preds=True)

    # mostly to test it works
    predictor.explain(save_shap_values=True)


def linear_nn(
    experiment="one_month_forecast", include_pred_month=True, surrounding_pixels=1
):
    predictor = LinearNetwork(
        layer_sizes=[100],
        data_folder=get_data_path(),
        experiment=experiment,
        include_pred_month=include_pred_month,
        surrounding_pixels=surrounding_pixels,
    )
    predictor.train(num_epochs=50, early_stopping=5)
    predictor.evaluate(save_preds=True)
    predictor.save_model()

    _ = predictor.explain(save_shap_values=True)


def rnn(experiment="one_month_forecast", include_pred_month=True, surrounding_pixels=1):
    predictor = RecurrentNetwork(
        hidden_size=128,
        data_folder=get_data_path(),
        experiment=experiment,
        include_pred_month=include_pred_month,
        surrounding_pixels=surrounding_pixels,
    )
    predictor.train(num_epochs=50, early_stopping=5)
    predictor.evaluate(save_preds=True)
    predictor.save_model()

    _ = predictor.explain(save_shap_values=True)


def earnn(
    experiment="one_month_forecast",
    include_pred_month=True,
    surrounding_pixels=None,
    pretrained=True,
):
    data_path = get_data_path()

    if not pretrained:
        predictor = EARecurrentNetwork(
            hidden_size=128,
            data_folder=data_path,
            experiment=experiment,
            include_pred_month=include_pred_month,
            surrounding_pixels=surrounding_pixels,
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
    parsimonious()
    regression()
    linear_nn()
    rnn()
    earnn(pretrained=True)
