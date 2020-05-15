"""

[d.name for d in (tmp_path / 'features').iterdir()]
"""

import numpy as np
from src.models.dynamic_data import DynamicDataLoader
import xarray as xr
from ..utils import _create_runoff_features_dir


class TestDynamicDataLoader:
    def test_dynamic_dataloader(self, tmp_path):
        x, y, static = _create_runoff_features_dir(tmp_path)
        # initialise the data as an OUTPUT of the engineers
        static_ignore_vars = ["area"]
        dynamic_ignore_vars = ["pet"]
        target_var = "discharge"
        seq_length = 5

        # initialize the object
        # dl = DynamicDataLoader(target_var=target_var,test_years=np.arange(2011, 2016),data_path=tmp_path,seq_length=seq_length,static_ignore_vars=static_ignore_vars,dynamic_ignore_vars=dynamic_ignore_vars,)
        dl = DynamicDataLoader(
            target_var=target_var,
            test_years=np.arange(2011, 2016),
            data_path=tmp_path,
            seq_length=seq_length,
            static_ignore_vars=static_ignore_vars,
            dynamic_ignore_vars=dynamic_ignore_vars,
        )
        X, y = dl.__iter__().__next__()

        assert isinstance(X, tuple)
        assert isinstance(X[0], np.ndarray)
        assert (
            X[0].shape[1] == seq_length
        ), "Dynamic data: Should be (#non-nan stations, seq_length, n_predictors)"
        assert (
            X[0].shape[-1] == 2
        ), "Dynamic data: Should be (#non-nan stations, seq_length, n_predictors)"

        assert isinstance(dl.static_ds, xr.Dataset)
        assert isinstance(dl.dynamic_ds, xr.Dataset)
        assert len(dl) == 16070
