import cdsapi
from pathlib import Path
import warnings

from typing import Dict

from .base import BaseExporter


class CDSExporter(BaseExporter):
    """Exports for the Climate Data Store

    cds.climate.copernicus.eu
    """

    def __init__(self, data_folder: Path = Path('data')) -> None:
        super().__init__(data_folder)
        self.client = cdsapi.Client()

    @staticmethod
    def make_filename(dataset: str, selection_request: Dict) -> str:
        """Makes the appropriate filename for a CDS export
        """
        date_str = ''
        if 'year' in selection_request:
            years = selection_request['year']
            if len(years) > 1:
                warnings.warn('More than 1 year of data being exported! '
                              'Export times may be significant.')
                years.sort()
                date_str = f'{years[0]}_{years[-1]}'
            else:
                date_str = str(years[0])
        elif 'date' in selection_request:
            date_str = selection_request['date']

        # force all data exports to be in netcdf format
        # TODO: This is not possible for some exports. We should select
        # our preferences and force those choices
        selection_request['format'] = 'netcdf'
        output_filename = f'{dataset.replace("-", "_")}_{date_str}.nc'
        return output_filename

    def export(self, dataset: str, selection_request: Dict) -> Path:
        """Export CDS data

        Parameters
        ----------
        dataset: str
            The dataset to be exported
        selection_request: dict
            The selection information to be passed to the CDS API

        Returns
        ----------
        output_file: Path
            The location of the exported data
        """

        output_filename = self.make_filename(dataset, selection_request)
        output_file = self.raw_folder / output_filename

        if not output_file.exists():
            self.client.retrieve(dataset, selection_request, str(output_file))

        return output_file
