import sys

sys.path.append("..")

from src.engineer import Engineer
from scripts.utils import get_data_path


def engineer(
    experiment="one_month_forecast",
    process_static=True,
    pred_months=12,
    target_variable: str = "VCI",
):

    engineer = Engineer(
        get_data_path(), experiment=experiment, process_static=process_static
    )
    engineer.engineer(
        test_year=[y for y in range(2015, 2019)],
        target_variable=target_variable,
        pred_months=pred_months,
        expected_length=pred_months,
    )


def engineer_static():
    Engineer.engineer_static_only(get_data_path())


if __name__ == "__main__":
    engineer(pred_months=5, process_static=True, target_variable="modis_vci")
    # engineer(pred_months=3, experiment="nowcast", process_static=True)
    # engineer_static()
