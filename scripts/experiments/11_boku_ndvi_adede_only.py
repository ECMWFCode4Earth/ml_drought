import sys
from pathlib import Path

sys.path.append("../..")

from scripts.utils import _rename_directory, get_data_path
from src.engineer import Engineer
from _base_models import parsimonious, regression, linear_nn, rnn, earnn
from scripts.experiments.adede_only_utils import rename_dirs, revert_interim_dirs


def engineer(pred_months=3, target_var="VCI1M"):
    engineer = Engineer(
        get_data_path(), experiment="one_month_forecast", process_static=False
    )
    engineer.engineer(
        test_year=[y for y in range(2016, 2019)],
        target_variable=target_var,
        pred_months=pred_months,
        expected_length=pred_months,
    )


def models(target_var: str = "VCI1M"):
    # NO IGNORE VARS
    ignore_vars = None
    # drop the target variable from ignore_vars
    # ignore_vars = [v for v in ignore_vars if v != target_var]
    # assert target_var not in ignore_vars

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
        to_path=data_path
        / "models"
        / f"one_month_forecast_BOKU_{target_var}_adede_only_vars",
    )


def move_features_dir(target_var):
    # rename the features dir
    data_path = get_data_path()

    _rename_directory(
        from_path=data_path / "features" / "one_month_forecast",
        to_path=data_path
        / "features"
        / f"one_month_forecast_BOKU_{target_var}_adede_only_vars",
    )


def main(monthly=True):
    # REQUIRES HAVING RUN preprocess() function in 10_boku_ndvi.py

    # assert False, "Need to work out why the LSTM / EALSTM is " \
    #     "predicting values upside down. See notebooks/draft/33_tl_..._.ipynb" \
    #     "somewhere we need to sort the latitude because they are predicting" \
    #     "values with the latitudes inverted ..."
    rename_dirs()

    target_vars = ["VCI1M", "VCI3M"]  #
    for target_var in target_vars:
        print(f"\n\n ** Target Variable: {target_var} ** \n\n")
        engineer(target_var=target_var)
        print(f"\n\n ** Running Models Target Variable: {target_var} ** \n\n")
        models(target_var=target_var)
        print(f"\n\n ** Models Run Target Variable: {target_var} ** \n\n")
        move_features_dir(target_var=target_var)

    revert_interim_dirs()


if __name__ == "__main__":
    main()
