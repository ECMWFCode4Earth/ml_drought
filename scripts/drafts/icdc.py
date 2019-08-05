from src.preprocess.icdc import (
    ESACCISoilMoisture,
    LAIModisAvhrr,
    ModisNDVI
)

processor = ModisNDVI()

processor.preprocess()
