import sys
from pathlib import Path

sys.path.append("../..")

# from src.exporters import BokuNDVIExporter
from src.preprocess import BokuNDVIPreprocessor

from scripts.utils import get_data_path

def main(monthly=True):
    regrid = get_data_path() / 'interim/VCI_preprocessed/data_kenya.nc'
    preprocessor = BokuNDVIPreprocessor(get_data_path(), resolution='1000')

    if monthly:
        preprocessor.preprocess(
            subset_str='kenya',
            regrid=regrid,
            resample_time='M'
        )
    else:
        preprocessor.preprocess(
            subset_str='kenya',
            regrid=regrid,
            resample_time='W-MON'
        )

if __name__ == "__main__":
    main()