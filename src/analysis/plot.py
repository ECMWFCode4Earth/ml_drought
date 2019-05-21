import xarray
from pathlib import Path


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
    def _cleanup(data: xarray.DataArray,
                 dropna: bool = True) -> xarray.DataArray:
        """Clean up the data so it is suitable for plotting

        Attributes:
        ----------
        data: xarray.DataArray
            A DataArray containing the data to plot
        dropna: bool, default = True
            Whether to drop nan values
        """
        if dropna:
            for dim in data.dims:
                data = data.dropna(dim=dim)

        return data
