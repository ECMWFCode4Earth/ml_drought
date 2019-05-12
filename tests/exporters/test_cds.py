from src.exporters.cds import CDSExporter, ERA5Exporter


class TestCDSExporter:

    def test_filename_year(self):

        dataset = 'megadodo-publications'
        selection_request = {
            'variable': ['towel'],
            'year': [1979, 1978, 1980]
        }

        filename = CDSExporter.make_filename(dataset, selection_request)
        expected = 'megadodo-publications_towel_1978_1980.nc'
        assert filename == expected, f'Got {filename}, expected {expected}!'

    def test_filename_date(self):
        dataset = 'megadodo-publications'
        selection_request = {
            'variable': ['towel'],
            'date': '1978-12-01/1980-12-31'
        }

        sanitized_date = selection_request["date"].replace('/', '_')
        filename = CDSExporter.make_filename(dataset, selection_request)
        expected = f'megadodo-publications_towel_{sanitized_date}.nc'
        assert filename == expected, f'Got {filename}, expected {expected}!'

    def test_selection_dict_granularity(self):

        selection_dict_monthly = ERA5Exporter.get_era5_times(granularity='monthly')
        assert 'day' not in selection_dict_monthly, 'Got day values in monthly the selection dict!'

        selection_dict_hourly = ERA5Exporter.get_era5_times(granularity='hourly')
        assert 'day' in selection_dict_hourly, 'Day values not in hourly selection dict!'

    def test_area(self):

        region = CDSExporter.get_kenya()
        kenya_str = CDSExporter.create_area(region)

        expected_str = '6.002/33.501/-5.202/42.283'
        assert kenya_str == expected_str, f'Got {kenya_str}, expected {expected_str}!'
