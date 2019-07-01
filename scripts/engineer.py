import sys
sys.path.append('..')

from pathlib import Path
from src.engineer import OneMonthForecastEngineer


def engineer():
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')

    engineer = OneMonthForecastEngineer(data_path)
    engineer.engineer(test_year=1994, target_variable='VHI')


if __name__ == '__main__':
    engineer()
