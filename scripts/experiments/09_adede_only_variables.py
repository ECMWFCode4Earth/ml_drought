import sys

sys.path.append("../..")

from _base_models import parsimonious, regression, linear_nn, rnn, earnn
from scripts.utils import _rename_directory, get_data_path
from src.engineer import Engineer

adede_vars = ["VCI1M", "VCI3M", "RFE1M", "RFE3M", "SPI1M", "SPI3M", "RCI1M", "RCI3M"]


def rename_dirs():
    data_path = get_data_path()

    # INTERIM
    if (data_path / "interim_adede_only").exists() and (data_path / "interim").exists():
        # move interim -> interim_
        # move interim_adede -> interim
        print("Moving data/interim -> data/interim_")
        _rename_directory(
            from_path=data_path / "interim",
            to_path=data_path / "interim_",
            with_datetime=False,
        )

        print("Moving data/interim_adede_only -> data/interim")
        _rename_directory(
            from_path=data_path / "interim_adede_only",
            to_path=data_path / "interim",
            with_datetime=False,
        )
    elif (
        not (data_path / "interim_adede_only").exists()
        and (data_path / "interim_").exists()
    ):
        # move interim_adede -> interim
        print("Moving data/interim_adede_only -> data/interim")
        _rename_directory(
            from_path=data_path / "interim_adede_only",
            to_path=data_path / "interim",
            with_datetime=False,
        )

    # check that correct dirs created
    assert not (data_path / "interim_adede_only").exists()
    assert (data_path / "interim").exists()
    assert (data_path / "interim_").exists()

    # FEATURES
    if (data_path / "features" / "one_month_forecast").exists():
        print(
            "Moving data/features/one_month_forecast -> data/features/one_month_forecast_"
        )
        _rename_directory(
            from_path=data_path / "features/one_month_forecast",
            to_path=data_path / "features/one_month_forecast_",
            with_datetime=False,
        )

    assert not (data_path / "features" / "one_month_forecast").exists()


def revert_interim_dirs():
    data_path = get_data_path()
    # INTERIM
    print("Moving data/interim -> data/interim_adede")
    _rename_directory(
        from_path=data_path / "interim",
        to_path=data_path / f"interim_adede_only",
        with_datetime=False,
    )
    print("Moving data/interim_ -> data/interim")
    _rename_directory(
        from_path=data_path / "interim_",
        to_path=data_path / "interim",
        with_datetime=False,
    )


def revert_features_dirs(target_var: str, original_dir: bool = False):
    data_path = get_data_path()
    # FEATURES
    print(
        f"Moving data/features/one_month_forecast -> data/features/one_month_forecast_adede_{target_var}"
    )
    _rename_directory(
        from_path=data_path / "features/one_month_forecast",
        to_path=data_path / f"features/one_month_forecast_adede_{target_var}",
        with_datetime=False,
    )

    if original_dir:
        print(
            "Moving data/features/one_month_forecast_ -> data/features/one_month_forecast"
        )
        _rename_directory(
            from_path=data_path / "features/one_month_forecast_",
            to_path=data_path / "features/one_month_forecast",
            with_datetime=False,
        )


# 2) ENGINEER the adede_vars to train/test
#    (VCI1M / VCI3M)
def engineer(pred_months=3, target_var="VCI1M"):
    engineer = Engineer(
        get_data_path(), experiment="one_month_forecast", process_static=False
    )
    engineer.engineer(
        test_year=[y for y in range(2011, 2019)],
        target_variable=target_var,
        pred_months=pred_months,
        expected_length=pred_months,
    )


# 3) run the models
def run_models(target_var: str):
    parsimonious()
    # -------
    # LSTM
    # -------
    rnn(  # earnn(
        experiment="one_month_forecast",
        include_pred_month=True,
        surrounding_pixels=None,
        explain=False,
        static=None,  # "features",
        ignore_vars=None,
        num_epochs=50,  # 50
        early_stopping=5,  # 5
        hidden_size=256,
        predict_delta=False,
        normalize_y=True,
        include_prev_y=False,
        include_latlons=False,
    )

    # -------
    # EALSTM
    # -------

    # rename the output file
    data_path = get_data_path()

    _rename_directory(
        from_path=data_path / "models" / "one_month_forecast",
        to_path=data_path
        / "models"
        / f"one_month_forecast_adede_only_target_{target_var}",
        with_datetime=False,
    )


if __name__ == "__main__":
    # 1) MOVE the current interim_adede_only and
    #    change interim_adede_only -> interim
    rename_dirs()

    for target_var in ["VCI1M", "VCI3M"]:
        print(f"\n\n** Running Experiment with {target_var} ** \n\n")
        engineer(target_var=target_var)
        run_models(target_var=target_var)
        revert_features_dirs(target_var=target_var)

    # change interim -> interim_adede_only
    revert_interim_dirs()
