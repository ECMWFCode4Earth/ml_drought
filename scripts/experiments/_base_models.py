import sys

sys.path.append("../..")
from src.models import (
    Persistence,
    LinearRegression,
    LinearNetwork,
    RecurrentNetwork,
    EARecurrentNetwork,
    load_model,
    GBDT,
)
from src.analysis import all_explanations_for_file
from scripts.utils import get_data_path


def parsimonious(experiment="one_month_forecast",):
    print("\n\n** Baseline **")
    predictor = Persistence(get_data_path(), experiment=experiment)
    predictor.evaluate(save_preds=True)
    print("\n\n")


def regression(
    experiment="one_month_forecast",
    include_pred_month=True,
    surrounding_pixels=None,
    explain=False,
    static="features",
    ignore_vars=None,
    predict_delta=False,
):
    print("\n\n** Regression **")
    predictor = LinearRegression(
        get_data_path(),
        experiment=experiment,
        include_pred_month=include_pred_month,
        surrounding_pixels=surrounding_pixels,
        static=static,
        ignore_vars=ignore_vars,
        predict_delta=predict_delta,
    )
    predictor.train()
    predictor.evaluate(save_preds=True)

    # mostly to test it works
    if explain:
        predictor.explain(save_shap_values=True)
    print("\n\n")


def linear_nn(
    experiment="one_month_forecast",
    include_pred_month=True,
    surrounding_pixels=None,
    explain=False,
    static="features",
    ignore_vars=None,
    num_epochs=50,
    early_stopping=5,
    layer_sizes=[100],
    predict_delta=False,
    learning_rate=1e-3,
):
    print("\n\n** Linear Network **")
    predictor = LinearNetwork(
        layer_sizes=layer_sizes,
        data_folder=get_data_path(),
        experiment=experiment,
        include_pred_month=include_pred_month,
        surrounding_pixels=surrounding_pixels,
        static=static,
        ignore_vars=ignore_vars,
        predict_delta=predict_delta,
    )
    predictor.train(
        num_epochs=num_epochs,
        early_stopping=early_stopping,
        learning_rate=learning_rate,
    )
    predictor.evaluate(save_preds=True)
    predictor.save_model()

    if explain:
        _ = predictor.explain(save_shap_values=True)
    print("\n\n")


def rnn(
    experiment="one_month_forecast",
    include_pred_month=True,
    surrounding_pixels=None,
    explain=False,
    static="features",
    ignore_vars=None,
    num_epochs=50,
    early_stopping=5,
    hidden_size=128,
    predict_delta=False,
    learning_rate=1e-3,
):
    print("\n\n** RNN **")
    predictor = RecurrentNetwork(
        hidden_size=hidden_size,
        data_folder=get_data_path(),
        experiment=experiment,
        include_pred_month=include_pred_month,
        surrounding_pixels=surrounding_pixels,
        static=static,
        ignore_vars=ignore_vars,
        predict_delta=predict_delta,
    )
    predictor.train(
        num_epochs=num_epochs,
        early_stopping=early_stopping,
        learning_rate=learning_rate,
    )
    predictor.evaluate(save_preds=True)
    predictor.save_model()

    if explain:
        _ = predictor.explain(save_shap_values=True)
    print("\n\n")


def earnn(
    experiment="one_month_forecast",
    include_pred_month=True,
    surrounding_pixels=None,
    pretrained=False,
    explain=False,
    static="features",
    ignore_vars=None,
    num_epochs=50,
    early_stopping=5,
    static_embedding_size=10,
    hidden_size=128,
    predict_delta=False,
    learning_rate=1e-3,
):
    print("\n\n** EALSTM **")
    data_path = get_data_path()

    if not pretrained:
        predictor = EARecurrentNetwork(
            hidden_size=hidden_size,
            data_folder=data_path,
            experiment=experiment,
            include_pred_month=include_pred_month,
            surrounding_pixels=surrounding_pixels,
            static=static,
            static_embedding_size=static_embedding_size,
            ignore_vars=ignore_vars,
            predict_delta=predict_delta,
        )
        predictor.train(
            num_epochs=num_epochs,
            early_stopping=early_stopping,
            learning_rate=learning_rate,
        )
        predictor.evaluate(save_preds=True)
        predictor.save_model()
    else:
        predictor = load_model(data_path / f"models/{experiment}/ealstm/model.pt")

    if explain:
        test_file = data_path / f"features/{experiment}/test/2018_3"
        assert test_file.exists()
        all_explanations_for_file(test_file, predictor, batch_size=100)
    print("\n\n")


def gbdt(
    experiment="one_month_forecast",
    include_pred_month=True,
    surrounding_pixels=None,
    pretrained=True,
    explain=False,
    static="features",
    ignore_vars=None,
    # predict_delta=False,
):
    print("\n\n** GBDT **")
    data_path = get_data_path()

    # initialise, train and save GBDT model
    predictor = GBDT(
        data_folder=data_path,
        experiment=experiment,
        include_pred_month=include_pred_month,
        surrounding_pixels=surrounding_pixels,
        static=static,
        ignore_vars=ignore_vars,
    )
    predictor.train(early_stopping=5)
    predictor.evaluate(save_preds=True)
    predictor.save_model()
    print("\n\n")
