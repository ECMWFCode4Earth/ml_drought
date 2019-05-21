import xarray
import numpy as np
from src.analysis.plot import Plotter


class TestPlotter:

    def test_make_analysis(self, tmp_path):

        dummy_dataset = xarray.Dataset()
        _ = Plotter(dummy_dataset, tmp_path)

        assert (tmp_path / 'analysis').exists(), 'Expected an analysis folder to be generated!'

    def test_summary(self):
        array = np.array([1, 2, 3, 4, 5])
        test_array = xarray.DataArray(array)

        summary = Plotter.calculate_summary(test_array, as_string=False)

        assert summary.min == array.min(), \
            f'Expected summary min to be {array.min()}, got {summary.min}'
        assert summary.max == array.max(), \
            f'Expected summary min to be {array.max()}, got {summary.max}'
        assert summary.mean == array.mean(), \
            f'Expected summary min to be {array.mean()}, got {summary.mean}'
        assert summary.median == np.median(array), \
            f'Expected summary min to be {np.median(array)}, got {summary.median}'
