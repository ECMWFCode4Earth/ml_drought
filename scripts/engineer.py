from pathlib import Path

import sys
sys.path.append('..')
from src.engineer import Engineer


def engineer():
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')

    engineer = Engineer(data_path)
    engineer.engineer(test_year=1994, target_variable='VHI',
                      target_month=6)


if __name__ == '__main__':
    engineer()
