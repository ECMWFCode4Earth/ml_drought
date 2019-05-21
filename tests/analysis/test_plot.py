import xarray
from src.analysis.plot import Plotter


class TestPlotter:

    def test_make_analysis(self, tmp_path):

        dummy_dataset = xarray.Dataset()
        _ = Plotter(dummy_dataset, tmp_path)

        assert (tmp_path / 'analysis').exists(), 'Expected an analysis folder to be generated!'
