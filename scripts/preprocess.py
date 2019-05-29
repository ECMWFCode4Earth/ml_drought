from pathlib import Path

import sys
sys.path.append('..')
from src.preprocess import VHIPreprocessor, CHIRPSPreprocesser  # noqa


def process_precip_2018():
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')
    processor = CHIRPSPreprocesser(data_path)

    processor.preprocess(subset_kenya=True,
                         regrid=None,
                         parallel=False)

def process_vhi_2018():
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')

    processor = VHIPreprocessor(data_path)




if __name__ == '__main__':
    process_precip_2018()
