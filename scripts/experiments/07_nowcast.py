import sys

sys.path.append("../..")

from scripts.utils import _rename_directory, get_data_path
from _base_models import parsimonious, regression, linear_nn, rnn, earnn
from scripts.engineer import engineer


def run_engineer() -> None:
    engineer(pred_months=12, experiment="nowcast")


if __name__ == "__main__":
    # NOTE: why have we downloaded 2 variables for ERA5 evaporaton
    # important_vars = ["VCI", "precip", "t2m", "pev", "p0005", "SMsurf", "SMroot"]
    # always_ignore_vars = ["ndvi", "p84.162", "sp", "tp", "Eb", "E", "p0001"]
    important_vars = ["precip", "t2m", "pev", "E", "SMsurf", "SMroot"]

    # IGNORE the forecast precip & VCI (to begin with)!
    ignore_vars = [
        "VCI",
        "ndvi",
        "p84.162",
        "sp",
        "tp",
        "Eb",
        "tprate_std_1",
        "tprate_mean_1",
        "tprate_std_2",
        "tprate_mean_2",
        "tprate_std_3",
        "tprate_mean_3",
    ]

    parsimonious(experiment="nowcast")
    # regression(ignore_vars=ignore_vars)
    # gbdt(ignore_vars=ignore_vars)
    # linear_nn(ignore_vars=ignore_vars)
    # rnn(ignore_vars=ignore_vars)
    earnn(
        experiment="nowcast",
        include_pred_month=True,
        surrounding_pixels=None,
        pretrained=False,
        explain=False,
        static="features",
        ignore_vars=ignore_vars,
        num_epochs=1,  #  50,
        early_stopping=5,
        hidden_size=256,
        static_embedding_size=64,
    )

    # rename the output file
    data_path = get_data_path()

    _rename_directory(
        from_path=data_path / "models" / "nowcast",
        to_path=data_path / "models" / "nowcast_tommy",
        with_datetime=True,
    )
