import sys

sys.path.append("../..")

from scripts.utils import _rename_directory, get_data_path
from _base_models import persistence, regression, linear_nn, rnn, earnn


if __name__ == "__main__":
    # NOTE: why have we downloaded 2 variables for ERA5 evaporaton
    # important_vars = ["VCI", "precip", "t2m", "pev", "p0005", "SMsurf", "SMroot"]
    # always_ignore_vars = ["ndvi", "p84.162", "sp", "tp", "Eb", "E", "p0001"]
    important_vars = ["VCI", "precip", "t2m", "pev", "E", "SMsurf", "SMroot"]
    always_ignore_vars = ["p84.162", "sp", "tp", "Eb", "VCI1M", "RFE1M"]  # "ndvi",

    # -------------
    # persistence
    # -------------
    persistence()

    # regression(ignore_vars=always_ignore_vars)
    # gbdt(ignore_vars=always_ignore_vars)
    # linear_nn(ignore_vars=always_ignore_vars)

    # -------------
    # LSTM
    # -------------
    rnn(
        experiment="one_month_forecast",
        include_pred_month=True,
        surrounding_pixels=None,
        explain=False,
        static="features",
        ignore_vars=always_ignore_vars,
        num_epochs=50,
        early_stopping=5,
        hidden_size=256,
        include_latlons=True,
    )

    # -------------
    # EALSTM
    # -------------
    earnn(
        experiment="one_month_forecast",
        include_pred_month=True,
        surrounding_pixels=None,
        pretrained=False,
        explain=False,
        static="features",
        ignore_vars=always_ignore_vars,
        num_epochs=50,
        early_stopping=5,
        hidden_size=256,
        static_embedding_size=64,
        include_latlons=True,
    )

    # rename the output file
    data_path = get_data_path()

    _rename_directory(
        from_path=data_path / "models" / "one_month_forecast",
        to_path=data_path / "models" / "one_month_forecast_BASE_adede_vars",
    )
