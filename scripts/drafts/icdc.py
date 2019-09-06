from src.preprocess.icdc import (
    ESACCISoilMoisture,
    LAIModisAvhrr,
    ModisNDVI
)


def modis_ndvi():
    processor = ModisNDVI()
    processor.preprocess()


def cci_soil_moisture():
    processor = ESACCISoilMoisture()
    processor.preprocess()


def modis_lai():
    processor = LAIModisAvhrr()
    processor.preprocess()


if __name__ == '__main__':
    modis_ndvi()
    cci_soil_moisture()
    modis_lai()
