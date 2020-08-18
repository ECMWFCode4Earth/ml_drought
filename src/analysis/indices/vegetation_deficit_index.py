from pathlib import Path
from typing import Optional
import xarray as xr
import numpy as np

from .base import BaseIndices
from .utils import rolling_mean


class VegetationDeficitIndex(BaseIndices):
    """3 month moving average VCI

    VCI3M Limits   | Description                        | Value
    -----------------------------------------------------------
    0 <= x <10     | Extreme vegetation deficit         |  1
    10 <= x <20    | Severe vegetation deficit          |  2
    20 <= x <35    | Moderate vegetation deficit        |  3
    35 <= x <50    | Normal vegetation conditions       |  4
    50 <= x <=100  | Above normal vegetation conditions |  5

    Klisch, A.; Atzberger, C.; Luminari, L. Satellite-Based Drought Monitoring
    In Kenya In An Operational Setting. Int. Arch. Photogramm. Remote Sens.
    Spat. Inf. Sci. 2015, XL-7/W3, 433–439

    Meroni, M.; Fasbender, D.; Rembold, F.; Atzberger, C.; Klisch, A. Near
    real-time vegetation anomaly detection with MODIS NDVI: Timeliness vs.
    accuracy and effect of anomaly computation options. Remote Sens. Environ.
    2019, 221, 508–521

    Klisch, A.; Atzberger, C. Operational drought monitoring in Kenya using
    MODIS NDVI time series. Remote Sens. 2016, 8, 267

    Adede, C., Oboko, R., Wagacha, P. W., & Atzberger, C. (2019). A mixed model
    approach to vegetation condition prediction using Artificial Neural Networks
    (ANN): Case of Kenya’s operational drought monitoring. Remote Sensing,
    11(9). https://doi.org/10.3390/rs11091099
    """

    def __init__(
        self,
        file_path: Path,
        rolling_window: int = 3,
        resample_str: Optional[str] = "month",
    ) -> None:

        if rolling_window != 3:
            print(
                "This index is fit on a 3 month VCI moving average."
                f" Are you sure you want to fit for {rolling_window}?"
            )

        self.name: str = "vegetation_deficit_index"
        self.ma_name: str = f"{rolling_window}{resample_str}_moving_average"
        self.rolling_window: int = rolling_window
        super().__init__(file_path=file_path, resample_str=resample_str)

        if "vci" not in [v.lower() for v in self.ds.data_vars]:
            print(
                "This is a VCI specific index. Are you sure you want to fit?"
                f" Found: {[v for v in self.ds.data_vars]}"
            )

    @staticmethod
    def vegetation_index_classify(
        da: xr.DataArray, new_variable_name: str
    ) -> xr.DataArray:
        """use the numpy `np.digitize` function to bin the
        values to their associated labels
        https://stackoverflow.com/a/56514582/9940782

        Arguments:
        ---------
        da : xr.DataArray
            variable that you want to bin into quintiles

        new_variable_name: str
            the `variable_name` in the output `xr.Dataset`
             which corresponds to the labels

        Returns:
        ------
        xr.Dataset

        Note:
            labels = [1, 2, 3, 4, 5] correspond to
             [(0, 10) (10, 20) (20, 35) (35, 50) (50, 100)]
        """
        # calculate the quintiles using `np.digitize`
        bins = [0.0, 10.0, 20.0, 35.0, 50.0]
        result = xr.apply_ufunc(np.digitize, da, bins)
        result = result.rename(new_variable_name)
        return result

    def fit(self, variable: str, time_str: str = "month") -> None:

        out_variable = f"VCI{self.rolling_window}M_index"
        print(f"Fitting {out_variable} for variable: {variable}")

        # 1. calculate a moving average
        ds_window = rolling_mean(self.ds, self.rolling_window)

        # calculate the VegetationIndex on moving average
        vci3m = self.vegetation_index_classify(ds_window[variable], out_variable)
        ds_window = ds_window.merge(vci3m.to_dataset(name=out_variable)).rename(
            {variable: f"{variable}{self.rolling_window}_moving_average"}
        )

        self.index = ds_window
        print("Fitted VegetationDeficitIndex and stored at `obj.index`")
