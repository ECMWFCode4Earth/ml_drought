import sys
from pathlib import Path
import datetime

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


<<<<<<< HEAD
def models(
    target_var: str = "boku_VCI",
    adede_only=False,
    experiment_name=None,
    check_inversion=False,
):
=======
def models(target_var: str = "boku_VCI", adede_only=False):
>>>>>>> f371ece16fee55a6fb6d7ab302ab79d11cb1a134
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
<<<<<<< HEAD
        ignore_vars = [
            "p84.162",
            "sp",
            "tp",
            "Eb",
            "VCI",
            "modis_ndvi",
            "SMroot",
            "SMsurf",
        ]
=======
        ignore_vars = ["p84.162", "sp", "tp", "Eb", "VCI", "modis_ndvi"]
>>>>>>> f371ece16fee55a6fb6d7ab302ab79d11cb1a134

    # drop the target variable from ignore_vars
    ignore_vars = [v for v in ignore_vars if v != target_var]
    assert target_var not in ignore_vars

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
<<<<<<< HEAD
        static="features",
        ignore_vars=ignore_vars,
        num_epochs=50,  # 1,  # 50 ,
        early_stopping=5,
        hidden_size=256,
        include_latlons=True,
        check_inversion=check_inversion,
=======
        static=None if adede_only else "features",
        ignore_vars=ignore_vars,
        num_epochs=50,
        early_stopping=5,
        hidden_size=256,
        include_latlons=True,
>>>>>>> f371ece16fee55a6fb6d7ab302ab79d11cb1a134
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
<<<<<<< HEAD
        static="features",
        ignore_vars=ignore_vars,
        num_epochs=50,  # 1,  # 50 ,
        early_stopping=5,
        hidden_size=256,
        static_embedding_size=64,
        include_latlons=True,
        check_inversion=check_inversion,
=======
        static=None if adede_only else "features",
        ignore_vars=ignore_vars,
        num_epochs=50,
        early_stopping=5,
        hidden_size=256,
        static_embedding_size=None if adede_only else 64,
        include_latlons=True,
>>>>>>> f371ece16fee55a6fb6d7ab302ab79d11cb1a134
    )

    # rename the output file
    data_path = get_data_path()
<<<<<<< HEAD
    if experiment_name is None:
        experiment_name = (
            f"one_month_forecast_BOKU_{target_var}_our_vars_{'only_P_VCI' if adede_only else 'ALL'}",
        )

    _rename_directory(
        from_path=data_path / "models" / "one_month_forecast",
        to_path=data_path / "models" / experiment_name,
    )


def move_features_dir(target_var, adede_only=False, experiment_name=None):
    # rename the features dir
    data_path = get_data_path()
    if experiment_name is None:
        experiment_name = f"one_month_forecast_BOKU_{target_var}_our_vars_{'only_P_VCI' if adede_only else 'ALL'}"

    _rename_directory(
        from_path=data_path / "features" / "one_month_forecast",
        to_path=data_path / "features" / experiment_name,
    )
=======

    _rename_directory(
        from_path=data_path / "models" / "one_month_forecast",
        to_path=data_path
        / "models"
        / f"one_month_forecast_BOKU_{target_var}_our_vars_{'only_P_VCI' if adede_only else 'ALL'}",
    )


def move_features_dir(target_var, adede_only=False):
    # rename the features dir
    data_path = get_data_path()
    try:
        _rename_directory(
            from_path=data_path / "features" / "one_month_forecast",
            to_path=data_path
            / "features"
            / f"one_month_forecast_BOKU_{target_var}_our_vars_{'only_P_VCI' if adede_only else 'ALL'}",
        )
    except Exception as E:
        print(E)
        date = datetime.datetime.now().strftime("%Y%M%d_%H%M")
        _rename_directory(
            from_path=data_path / "features" / "one_month_forecast",
            to_path=data_path
            / "features"
            / f"one_month_forecast_BOKU_{target_var}_our_vars_{date}",
        )
>>>>>>> f371ece16fee55a6fb6d7ab302ab79d11cb1a134


def main(monthly=True):
    # preprocess(monthly=monthly)
<<<<<<< HEAD
    ADEDE_ONLY = False
    TARGET_VARS = ["boku_VCI", "VCI3M"]  # "boku_VCI", "VCI3M"

    for target_var in TARGET_VARS:
        print(f"\n\n ** Target Variable: {target_var} ** \n\n")
        engineer(target_var=target_var)
        print(f"\n\n ** RUNNING MODELS FOR Target Variable: {target_var} ** \n\n")
        models(
            target_var=target_var,
            adede_only=ADEDE_ONLY,
            experiment_name=None,
            check_inversion=True,
        )
        print(f"\n\n ** Target Variable: {target_var} DONE ** \n\n")
        # move_features_dir(target_var=target_var, adede_only=adede_only)
=======

    adede_only = True
    target_vars = ["VCI3M"]  # "boku_VCI",
    for target_var in target_vars:
        print(f"\n\n ** Target Variable: {target_var} ** \n\n")
        engineer(target_var=target_var)
        print(f"\n\n ** RUNNING MODELS FOR Target Variable: {target_var} ** \n\n")
        models(target_var=target_var, adede_only=adede_only)
        print(f"\n\n ** Target Variable: {target_var} DONE ** \n\n")
        move_features_dir(target_var=target_var, adede_only=adede_only)
>>>>>>> f371ece16fee55a6fb6d7ab302ab79d11cb1a134


if __name__ == "__main__":
    main()
