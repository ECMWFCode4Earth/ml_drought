import boto3
import botocore
from botocore.client import Config
from itertools import product
from pathlib import Path
import warnings


from typing import List, Optional

from .base import BaseExporter


class ERA5ExporterPOS(BaseExporter):
    """Download ERA5 data from Planet OS's S3 bucket
    """

    def __init__(self, data_folder: Path = Path('data')) -> None:
        super().__init__(data_folder)

        self.era5_folder = self.raw_folder / 'era5POS'
        if not self.era5_folder.exists():
            self.era5_folder.mkdir()

        self.era5_bucket = 'era5-pds'
        self.client = boto3.client('s3', config=Config(signature_version=botocore.UNSIGNED))

    def _get_available_years(self) -> List[int]:
        """Currently, data is only available from 2008, with a
        view to extend this to 1979.

        This method allows the exporter to identify how much data is
        currently available in the bucket

        Returns
        ----------
        years: A list of years for which data is available
        """
        years = []
        paginator = self.client.get_paginator('list_objects')
        result = paginator.paginate(Bucket=self.era5_bucket, Delimiter='/')
        for prefix in result.search('CommonPrefixes'):
            try:
                # the buckets have a backslash at the end which we want to
                # remove
                years.append(int(prefix.get('Prefix')[:-1]))
            except ValueError:
                # one of the buckets is QA/
                continue
        return years

    def export(self,
               variable: str,
               years: Optional[List[int]] = None,
               months: Optional[List[int]] = None) -> List[Path]:

        if years is None:
            years = self._get_available_years()

        if months is None:
            months = list(range(1, 12 + 1))

        output_files = []
        for year, month in product(years, months):
            target_key = f'{year}/{month:02d}/data/{variable}.nc'

            target_folder = self.era5_folder / f'{year}/{month:02d}'
            target_folder.mkdir(parents=True, exist_ok=True)
            target_output = target_folder / f'{variable}.nc'
            try:
                self.client.download_file(self.era5_bucket,
                                          target_key,
                                          str(target_output))
                output_files.append(target_output)
            except botocore.exceptions.ClientError as e:
                if e.response['Error']['Code'] == "404":
                    warnings.warn('Key does not exist!')
                else:
                    raise e
        return output_files
