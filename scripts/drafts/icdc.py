from src.preprocess.icdc import (
    ESACCISoilMoisture,
    LAIModisAvhrr,
    ModisNDVI
)

processor = ModisNDVI()
# processor.preprocess()

subset_str='kenya'
resample_time='M'
upsampling=False
processor.merge_files(subset_str, resample_time, upsampling)
