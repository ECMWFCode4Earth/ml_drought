import sys
from pathlib import Path

sys.path.append("../..")

# from src.exporters import BokuNDVIExporter
from src.preprocess import BokuNDVIPreprocessor

from scripts.utils import get_data_path
from src.engineer import Engineer

def preprocess(monthly=True):
    regrid = get_data_path() / "interim/VCI_preprocessed/data_kenya.nc"
    preprocessor = BokuNDVIPreprocessor(get_data_path(), resolution="1000")

    if monthly:
        preprocessor.preprocess(subset_str="kenya", regrid=regrid, resample_time="M")
    else:
        preprocessor.preprocess(
            subset_str="kenya", regrid=regrid, resample_time="W-MON"
        )


def engineer(pred_months=3, target_var="boku_VCI"):
    engineer = Engineer(
        get_data_path(), experiment="one_month_forecast", process_static=False
    )
    engineer.engineer(
        test_year=[y for y in range(2016, 2019)],
        target_variable=target_var,
        pred_months=pred_months,
        expected_length=pred_months,
    )


def main(monthly=True):
    # preprocess(monthly=monthly)
    engineer()
    # ignore_vars = ["p84.162", "sp", "tp", "Eb", "modis_ndvi", "VCI1M", "RFE1M"]  # "ndvi",


if __name__ == "__main__":
    main()
