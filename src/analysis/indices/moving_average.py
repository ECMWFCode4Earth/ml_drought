from pathlib import Path
from typing import Optional

from .base import BaseIndices
from .utils import rolling_mean


class MovingAverage(BaseIndices):
    def __init__(
        self,
        file_path: Path,
        rolling_window: int = 3,
        resample_str: Optional[str] = "month",
    ) -> None:

        self.name = f"{rolling_window}{resample_str}_moving_average"
        self.rolling_window = rolling_window
        super().__init__(file_path=file_path, resample_str=resample_str)

    def fit(self, variable: str, time_str: str = "month") -> None:
        vars = [v for v in self.ds.data_vars]
        assert (
            variable in vars
        ), f"Must apply MovingAverage to a \
        variable in `self.ds`: {vars}"

        print(f"Fitting {self.name} for variable: {variable}")
        # 1. calculate a moving average
        ds_window = rolling_mean(self.ds, self.rolling_window)

        # 2. calculate the moving average for the variable
        out_variable_name = f"{variable}_{self.name}"
        ds_window = ds_window.rename({variable: out_variable_name})

        # 3. return fitted moving average
        self.index = ds_window
        print(f"Fitted {self.name} and stored at `obj.index`")
