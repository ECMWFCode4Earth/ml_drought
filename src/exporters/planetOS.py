from itertools import product
from pathlib import Path
import warnings

from typing import List, Optional

from .base import BaseExporter

boto3 = None
botocore = None
Config = None


class ERA5ExporterPOS(BaseExporter):
    """Download ERA5 data from Planet OS's S3 bucket

    https://github.com/planet-os/notebooks/blob/master/aws/era5-pds.md
    """

    dataset = "era5POS"

    def __init__(self, data_folder: Path = Path("data")) -> None:
        super().__init__(data_folder)

        global boto3
        if boto3 is None:
            import boto3
        global botocore
        global Config
        if botocore is None:
            import botocore
            from botocore.client import Config

        self.era5_bucket = "era5-pds"
        self.client = boto3.client(  # type: ignore
            "s3",
            config=Config(  # type: ignore
                signature_version=botocore.UNSIGNED
            ),
        )

    def get_variables(self, year: int, month: int) -> List[str]:
        target_prefix = f"{year}/{month:02d}/data"

        files = self.client.list_objects_v2(
            Bucket=self.era5_bucket, Prefix=target_prefix
        )
        variables = []
        for file in files["Contents"]:
            key = file["Key"].split("/")[-1].replace(".nc", "")
            variables.append(key)

        return variables

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
        paginator = self.client.get_paginator("list_objects")
        result = paginator.paginate(Bucket=self.era5_bucket, Delimiter="/")
        for prefix in result.search("CommonPrefixes"):
            try:
                # the buckets have a backslash at the end which we want to
                # remove
                years.append(int(prefix.get("Prefix")[:-1]))
            except ValueError:
                # one of the buckets is QA/
                continue
        return years

    def export(
        self,
        variable: str,
        years: Optional[List[int]] = None,
        months: Optional[List[int]] = None,
    ) -> List[Path]:
        """Export data from Planet OS's S3 bucket

        Arguments
        ----------
        variable: str
            The variable to download
        years: list of ints or None, default = None
            The years of data to download
        months: list of ints, or None, default = None
            The months of data to download

        Returns
        ----------
        output_files: list of pathlib.Paths
            The locations of the saved files
        """

        if years is None:
            years = self._get_available_years()

        if months is None:
            months = list(range(1, 12 + 1))

        output_files = []
        for year, month in product(years, months):
            target_key = f"{year}/{month:02d}/data/{variable}.nc"

            target_folder = self.output_folder / f"{year}/{month:02d}"
            target_folder.mkdir(parents=True, exist_ok=True)
            target_output = target_folder / f"{variable}.nc"

            if target_output.exists():
                print(f"{target_output} already exists! Skipping")
                continue

            try:
                self.client.download_file(
                    self.era5_bucket, target_key, str(target_output)
                )
                output_files.append(target_output)
                print(f"Exported {target_key} to {target_folder}")
            except botocore.exceptions.ClientError as e:  # type: ignore
                if e.response["Error"]["Code"] == "404":
                    possible_variables = self.get_variables(year, month)
                    possible_variables_str = "\n".join(possible_variables)
                    warnings.warn(
                        f"Key does not exist! "
                        f"Possible variables are: {possible_variables_str}"
                    )
                else:
                    raise e
        return output_files
