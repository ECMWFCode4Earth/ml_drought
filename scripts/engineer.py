from pathlib import Path
import sys

sys.path.append("..")

from src.engineer import Engineer


def engineer(experiment="one_month_forecast", process_static=True, pred_months=12):
    # if the working directory is alread ml_drought don't need ../data
    if Path(".").absolute().as_posix().split("/")[-1] == "ml_drought":
        data_path = Path("data")
    else:
        data_path = Path("../data")

    engineer = Engineer(data_path, experiment=experiment, process_static=process_static)
    engineer.engineer(
        test_year=2018,
        target_variable="VCI",
        pred_months=pred_months,
        expected_length=pred_months,
    )


def engineer_static():
    # if the working directory is alread ml_drought don't need ../data
    if Path(".").absolute().as_posix().split("/")[-1] == "ml_drought":
        data_path = Path("data")
    else:
        data_path = Path("../data")
    Engineer.engineer_static_only(data_path)


if __name__ == "__main__":
    engineer(pred_months=12)
    # engineer_static()
