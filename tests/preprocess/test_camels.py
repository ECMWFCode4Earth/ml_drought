from ..utils import _copy_runoff_data_to_tmp_path
from src.preprocess import CAMELSGBPreprocessor
import numpy as np
import xarray as xr


class TestCAMELSGBPreprocessor:
    def test(self, tmp_path):
        _copy_runoff_data_to_tmp_path(tmp_path)
        processor = CAMELSGBPreprocessor(tmp_path, open_shapefile=False)
        processor.preprocess()

        assert isinstance(processor, CAMELSGBPreprocessor)

        # check folder structure created
        assert all(np.isin(["interim", "raw"], [d.name for d in tmp_path.iterdir()]))
        assert all(
            np.isin(
                ["camels_preprocessed", "static"],
                [d.name for d in (tmp_path / "interim").iterdir()],
            )
        )
        assert (tmp_path / "interim/static/data.nc").exists()
        assert (tmp_path / "interim/camels_preprocessed/data.nc").exists()

        # check the preprocessed data
        static = xr.open_dataset((tmp_path / "interim/static/data.nc"))
        dynamic = xr.open_dataset((tmp_path / "interim/camels_preprocessed/data.nc"))

        assert all(np.isin(["time", "station_id"], [c for c in dynamic.coords]))
        assert all(np.isin(["station_id"], [c for c in static.coords]))
