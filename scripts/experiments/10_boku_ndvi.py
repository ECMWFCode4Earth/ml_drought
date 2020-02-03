import sys
from pathlib import Path

sys.path.append("../..")

# from src.exporters import BokuNDVIExporter
from src.preprocess import BokuNDVIPreprocessor

from scripts.utils import _rename_directory, get_data_path
from src.engineer import Engineer
from _base_models import parsimonious, regression, linear_nn, rnn, earnn


def preprocess(monthly=True):
    regrid = get_data_path() / "interim/VCI_preprocessed/data_kenya.nc"
    preprocessor = BokuNDVIPreprocessor(get_data_path(), resolution="1000")

    if monthly:
        preprocessor.preprocess(subset_str="kenya", regrid=regrid, resample_time="M")
    else:
        preprocessor.preprocess(
            subset_str="kenya", regrid=regrid, resample_time="W-MON"
        )


def engineer(pred_months=3, target_var="boku_VCI"):
    engineer = Engineer(
        get_data_path(), experiment="one_month_forecast", process_static=False
    )
    engineer.engineer(
        test_year=[y for y in range(2016, 2019)],
        target_variable=target_var,
        pred_months=pred_months,
        expected_length=pred_months,
    )


def models():
    ignore_vars = ["p84.162", "sp", "tp", "Eb", "VCI", "boku_VCI"]  # modis_ndvi
    # -------------
    # persistence
    # -------------
    parsimonious()

    # regression(ignore_vars=ignore_vars)
    # gbdt(ignore_vars=ignore_vars)
    # linear_nn(ignore_vars=ignore_vars)

    # -------------
    # LSTM
    # -------------
    rnn(
        experiment="one_month_forecast",
        include_pred_month=True,
        surrounding_pixels=None,
        explain=False,
        static="features",
        ignore_vars=ignore_vars,
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
        ignore_vars=ignore_vars,
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
        to_path=data_path / "models" / "one_month_forecast_boku_ndvi",
    )


def main(monthly=True):
    # preprocess(monthly=monthly)
    engineer(target_var="modis_ndvi")
    models()


if __name__ == "__main__":
    main()
