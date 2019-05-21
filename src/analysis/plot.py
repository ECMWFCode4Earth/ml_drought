import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import seaborn as sns
from pathlib import Path
from collections import namedtuple
import xarray

from typing import Union, Optional


Summary = namedtuple('Summary', ['min', 'max', 'mean', 'median'])


class Plotter:
    """Base for all plotting functionality.

    Attributes:
    ----------
    data: xarray.Dataset
        A dataset containing the data to plot
    data_folder: pathlib.Path, default: Path('data')
        The data folder into which results of the analysis will be saved
    """
    def __init__(self, data: xarray.Dataset,
                 data_folder: Path = Path('data')) -> None:
        self.raw_data = data
        self.data_folder = data_folder
        self.analysis_folder = self.data_folder / 'analysis'
        if not self.analysis_folder.exists():
            self.analysis_folder.mkdir()

    @staticmethod
    def calculate_summary(data_array: xarray.DataArray,
                          as_string: bool = True) -> Union[str, Summary]:
        summary = Summary(
            min=data_array.min(),
            max=data_array.max(),
            mean=data_array.mean(),
            median=np.median(data_array)
        )
        if as_string:
            strings = [': '.join([key, f'{val:.2f}']) for key, val in summary._asdict().items()]

            return ', '.join(strings)
        return summary

    def plot_histogram(self, variable: str,
                       save: bool = True,
                       add_summary: bool = True,
                       title: Optional[str] = None,
                       ax: Optional[plt.axes] = None,
                       return_axes: bool = False) -> Optional[plt.axes]:
        """Plot a histogram
        """
        assert variable in self.raw_data, f'{variable} not a variable in the dataset'
        data_array = self.raw_data[variable]

        data_array = data_array.values[~np.isnan(data_array.values)]
        units = self.raw_data[variable].units

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))

        # plot the histogram
        sns.distplot(data_array, ax=ax)

        if title is None:
            title = f'Density Plot of {variable}'
            if add_summary:
                summary = self.calculate_summary(data_array, as_string=True)
                title = f'{title}\n{summary}'

        ax.set_title(title)
        ax.set_xlabel(units)

        if save:
            filename = f'{datetime.now().date()}_{variable}_histogram'
            print(f'Saving to {self.analysis_folder / filename}')
            plt.savefig(self.analysis_folder / filename, bbox_inches='tight', dpi=300)

        if return_axes:
            return ax
        return None
