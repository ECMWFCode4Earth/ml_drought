import sys

sys.path.append("..")

from src.engineer import Engineer
from scripts.utils import get_data_path
from src.models import (
    Persistence,
    LinearRegression,
    LinearNetwork,
    RecurrentNetwork,
    EARecurrentNetwork,
    load_model,
)
from scripts.models import (
    persistence,
    regression,
    linear_nn,
    rnn,
    earnn,
    get_forecast_vars,
    get_ignore_static_vars,
)


def engineer(
    experiment="one_month_forecast",
    process_static=True,
    pred_months=12,
    test_years=[y for y in range(2011, 2019)],
):

    engineer = Engineer(
        get_data_path(), experiment=experiment, process_static=process_static
    )
    engineer.engineer(
        test_year=[y for y in range(2015, 2019)],
        target_variable="VCI",
        pred_months=pred_months,
        expected_length=pred_months,
    )


def model():
    forecast_vars = get_forecast_vars()
    ignore_static_vars = get_ignore_static_vars()
    ignore_vars = forecast_vars + ignore_static_vars

    persistence(experiment="nowcast")
    regression(experiment="nowcast", ignore_vars=ignore_vars)
    linear_nn(experiment="nowcast", ignore_vars=ignore_vars, static=None)
    rnn(experiment="nowcast", ignore_vars=ignore_vars, static=None)
    earnn(experiment="nowcast", ignore_vars=ignore_vars, static="features")

    pass


if __name__ == "__main__":
    engineer(experiment="nowcast", pred_months=3, process_static=True)
    model()
