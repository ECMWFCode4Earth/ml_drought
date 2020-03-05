import sys

sys.path.append("..")

from src.engineer import Engineer
from scripts.utils import get_data_path


def engineer(experiment="one_month_forecast", process_static=True, seq_length=12):

    engineer = Engineer(
        get_data_path(), experiment=experiment, process_static=process_static
    )
    engineer.engineer(
        test_year=[y for y in range(2011, 2019)],
        target_variable="VCI",
        seq_length=seq_length,
        expected_length=seq_length,
    )


def engineer_static():
    Engineer.engineer_static_only(get_data_path())


if __name__ == "__main__":
    engineer(seq_length=3)
    # engineer(seq_length=12, experiment='nowcast')
    # engineer_static()
