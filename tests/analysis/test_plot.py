import xarray
import numpy as np
from datetime import datetime
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

    def test_histogram_plot(self, tmp_path):

        dataset = xarray.Dataset({'VHI': [1, 2, 3, 4]})
        plotter = Plotter(dataset, tmp_path)

        hist_ax = plotter.plot_histogram('VHI', add_summary=False, return_axes=True)

        today = datetime.now().date()

        expected_file_location = tmp_path / 'analysis' / f'{today}_VHI_histogram.png'
        assert expected_file_location.exists(), \
            f'Expected histogram to be saved at {expected_file_location}'

        expected_title = 'Density Plot of VHI'
        assert hist_ax.title.get_text() == expected_title, \
            f'Expected image title to be {expected_title}, got {hist_ax.title.get_text()}'
