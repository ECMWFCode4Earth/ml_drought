import sys
from pathlib import Path
import datetime

sys.path.append("../..")

# from src.exporters import BokuNDVIExporter
# from src.preprocess import BokuNDVIPreprocessor

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


def models(
    target_var: str = "boku_VCI",
    adede_only=False,
    rename: bool = True,
    ealstm: bool = True,
    lstm: bool = True,
    baseline: bool = True,
):
    if adede_only:
        ignore_vars = [
            "p84.162",
            "sp",
            "tp",
            "Eb",
            "VCI",
            "modis_ndvi",
            "pev",
            "t2m",
            "E",
            "SMroot",
            "SMsurf",
        ]
    else:
        ignore_vars = ["p84.162", "sp", "tp", "Eb", "VCI", "modis_ndvi"]

    # drop the target variable from ignore_vars
    ignore_vars = [v for v in ignore_vars if v != target_var]
    assert target_var not in ignore_vars

    # -------------
    # persistence
    # -------------
    if baseline:
        parsimonious(include_yearly_aggs=False)

    # regression(ignore_vars=ignore_vars)
    # gbdt(ignore_vars=ignore_vars)
    # linear_nn(ignore_vars=ignore_vars)

    # -------------
    # LSTM
    # -------------
    if lstm:
        rnn(
            experiment="one_month_forecast",
            include_pred_month=True,
            surrounding_pixels=None,
            explain=False,
            static=None if adede_only else "features",
            ignore_vars=ignore_vars,
            num_epochs=50,
            early_stopping=5,
            hidden_size=256,
            include_latlons=True,
            include_yearly_aggs=False,
        )

    # -------------
    # EALSTM
    # -------------
    if ealstm:
        earnn(
            experiment="one_month_forecast",
            include_pred_month=True,
            surrounding_pixels=None,
            pretrained=False,
            explain=False,
            static=None if adede_only else "features",
            ignore_vars=ignore_vars,
            num_epochs=50,
            early_stopping=5,
            hidden_size=256,
            static_embedding_size=None if adede_only else 64,
            include_latlons=True,
            include_yearly_aggs=False,
        )

    # rename the output file
    if rename:
        data_path = get_data_path()

        _rename_directory(
            from_path=data_path / "models" / "one_month_forecast",
            to_path=data_path
            / "models"
            / f"ICLR_one_month_forecast_BOKU_{target_var}_our_vars_{'only_P_VCI' if adede_only else 'ALL'}",
        )


def move_features_dir(target_var, adede_only=False):
    # rename the features dir
    data_path = get_data_path()
    try:
        _rename_directory(
            from_path=data_path / "features" / "one_month_forecast",
            to_path=data_path
            / "features"
            / f"ICLR_one_month_forecast_BOKU_{target_var}_our_vars_{'only_P_VCI' if adede_only else 'ALL'}",
        )
    except Exception as E:
        print(E)
        date = datetime.datetime.now().strftime("%Y%M%d_%H%M")
        _rename_directory(
            from_path=data_path / "features" / "one_month_forecast",
            to_path=data_path
            / "features"
            / f"ICLR_one_month_forecast_BOKU_{target_var}_our_vars_{date}",
        )


def main(monthly=True):
    # preprocess(monthly=monthly)

    adede_only = False
    rename = False
    target_vars = ["boku_VCI"]  # "boku_VCI", "VCI3M"
    for target_var in target_vars:
        print(f"\n\n ** Target Variable: {target_var} ** \n\n")
        # engineer(target_var=target_var)

        print(f"\n\n ** RUNNING MODELS FOR Target Variable: {target_var} ** \n\n")
        models(
            target_var=target_var,
            adede_only=adede_only,
            rename=rename,
            ealstm=True,
            lstm=False,
            baseline=False,
        )

        print(f"\n\n ** Target Variable: {target_var} DONE ** \n\n")
        # move_features_dir(target_var=target_var, adede_only=adede_only)


if __name__ == "__main__":
    main()
