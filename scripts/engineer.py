import sys
sys.path.append('..')

from pathlib import Path
from src.engineer import OneMonthForecastEngineer, NowcastEngineer


def engineer(pred_months=11):
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')

    engineer = OneMonthForecastEngineer(data_path)
    engineer.engineer(
        test_year=2018, target_variable='VHI',
        pred_months=pred_months, expected_length=pred_months,
    )


def nowcast_engineer():
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')

    engineer = NowcastEngineer(data_path)
    engineer.engineer(test_year=2018, target_variable='VHI')


if __name__ == '__main__':
    engineer(pred_months=3)
    nowcast_engineer(pred_months=3)
