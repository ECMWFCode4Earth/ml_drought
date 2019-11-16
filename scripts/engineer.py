import sys

sys.path.append("..")

from src.engineer import Engineer
from scripts.utils import get_data_path


def engineer(experiment="one_month_forecast", process_static=True, pred_months=12):

    engineer = Engineer(get_data_path(), experiment=experiment, process_static=process_static)
    engineer.engineer(
        test_year=2018,
        target_variable="VCI",
        pred_months=pred_months,
        expected_length=pred_months,
    )


def engineer_static():
    Engineer.engineer_static_only(get_data_path())


if __name__ == "__main__":
    engineer(pred_months=12)
    # engineer_static()
