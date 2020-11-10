import sys

sys.path.append("..")

from src.engineer import engineer
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


def model():
    forecast_vars = get_forecast_vars()
    ignore_static_vars = get_ignore_static_vars()
    ignore_vars = forecast_vars + ignore_static_vars

    persistence(experiment="nowcast")
    regression(experiment="nowcast", ignore_vars=ignore_vars)
    linear_nn(experiment="nowcast", ignore_vars=ignore_vars, static="features")
    rnn(experiment="nowcast", ignore_vars=ignore_vars, static="features")
    earnn(experiment="nowcast", ignore_vars=ignore_vars, static="features")

    pass


if __name__ == "__main__":
    engineer(experiment="nowcast", pred_months=3, process_static=True)
    model()
