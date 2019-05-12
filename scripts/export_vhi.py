from pathlib import Path

import sys
sys.path.append('..')
from src.exporters import CDSExporter  # noqa


def export_vhi_2018():
    data_path = Path('../data')
    exporter = CDSExporter(data_path)

    selection_request = exporter.get_era5_times(granularity='hourly')
    # we only want to download 2018 data
    selection_request['year'] = ['2018']

    dataset = 'reanalysis-era5-single-levels'
    variables = ['high_vegetation_cover',
                 'leaf_area_index_high_vegetation',
                 'leaf_area_index_low_vegetation',
                 'low_vegetation_cover',
                 'type_of_high_vegetation',
                 'type_of_low_vegetation']

    # add the other two necessary arguments to the selection dict
    selection_request['variable'] = variables
    kenya_region = exporter.get_kenya()
    selection_request['area'] = exporter.create_area(kenya_region)

    exporter.export(dataset, selection_request)


if __name__ == '__main__':
    export_vhi_2018()
