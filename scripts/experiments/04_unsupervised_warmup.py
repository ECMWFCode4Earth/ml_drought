import sys

sys.path.append("../..")

from scripts.utils import get_data_path, _rename_directory
from src.analysis import all_shap_for_file
from src.models import (
    LinearNetwork,
    RecurrentNetwork,
    EARecurrentNetwork,
    load_model,
)


def linear_nn(
    experiment="one_month_forecast",
    include_pred_month=True,
    surrounding_pixels=1,
    explain=False,
    static="features",
    ignore_vars=None,
):
    predictor = LinearNetwork(
        layer_sizes=[100],
        data_folder=get_data_path(),
        experiment=experiment,
        include_pred_month=include_pred_month,
        surrounding_pixels=surrounding_pixels,
        static=static,
        ignore_vars=ignore_vars,
    )
    predictor.unsupervised_warm_up(
        num_epochs=50, early_stopping=5, neighbouring_distance=2, multiplier=10
    )
    predictor.train(num_epochs=50, early_stopping=5)
    predictor.evaluate(save_preds=True)
    predictor.save_model()

    if explain:
        _ = predictor.explain(save_shap_values=True)


def rnn(
    experiment="one_month_forecast",
    include_pred_month=True,
    surrounding_pixels=1,
    explain=False,
    static="features",
    ignore_vars=None,
):
    predictor = RecurrentNetwork(
        hidden_size=128,
        data_folder=get_data_path(),
        experiment=experiment,
        include_pred_month=include_pred_month,
        surrounding_pixels=surrounding_pixels,
        static=static,
        ignore_vars=ignore_vars,
    )
    predictor.unsupervised_warm_up(
        num_epochs=50, early_stopping=5, neighbouring_distance=2, multiplier=10
    )
    predictor.train(num_epochs=50, early_stopping=5)
    predictor.evaluate(save_preds=True)
    predictor.save_model()

    if explain:
        _ = predictor.explain(save_shap_values=True)


def earnn(
    experiment="one_month_forecast",
    include_pred_month=True,
    surrounding_pixels=None,
    pretrained=False,
    explain=False,
    static="features",
    ignore_vars=None,
):
    data_path = get_data_path()

    if not pretrained:
        predictor = EARecurrentNetwork(
            hidden_size=128,
            data_folder=data_path,
            experiment=experiment,
            include_pred_month=include_pred_month,
            surrounding_pixels=surrounding_pixels,
            static=static,
            static_embedding_size=10,
            ignore_vars=ignore_vars,
        )
        predictor.unsupervised_warm_up(
            num_epochs=50, early_stopping=5, neighbouring_distance=2, multiplier=10
        )
        predictor.train(num_epochs=50, early_stopping=5)
        predictor.evaluate(save_preds=True)
        predictor.save_model()
    else:
        predictor = load_model(data_path / f"models/{experiment}/ealstm/model.pt")

    if explain:
        test_file = data_path / f"features/{experiment}/test/2018_3"
        assert test_file.exists()
        all_shap_for_file(test_file, predictor, batch_size=100)


if __name__ == "__main__":
    # NOTE: why have we downloaded 2 variables for ERA5 evaporaton
    # important_vars = ["VCI", "precip", "t2m", "pev", "p0005", "SMsurf", "SMroot"]
    # always_ignore_vars = ["ndvi", "p84.162", "sp", "tp", "Eb", "E", "p0001"]
    important_vars = ["VCI", "precip", "t2m", "pev", "E", "SMsurf", "SMroot"]
    always_ignore_vars = ["ndvi", "p84.162", "sp", "tp", "Eb"]

    # linear_nn(ignore_vars=always_ignore_vars)
    # rnn(ignore_vars=always_ignore_vars)
    earnn(ignore_vars=always_ignore_vars)

    # rename the output file
    data_path = get_data_path()

    _rename_directory(
        from_path=data_path / "models" / "one_month_forecast",
        to_path=data_path / "models" / "one_month_forecast_unsupervised_warmup",
    )
