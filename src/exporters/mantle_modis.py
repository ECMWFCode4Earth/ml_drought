from pathlib import Path
import numpy as np
import pandas as pd
from itertools import product
from typing import List, Dict, Optional, Any
import warnings
from tqdm import tqdm
from .base import BaseExporter
import shutil
import xarray as xr

gdal = None
boto3 = None
botocore = None
Config = None

DEFAULT_ARGS = {
    "region_name": "eu-central-1",
    "api_version": None,
    "use_ssl": True,
    "verify": None,
    "endpoint_url": None,
    "aws_access_key_id": None,
    "aws_secret_access_key": None,
    "aws_session_token": None,
    "config": None,
}


class MantleModisExporter(BaseExporter):
    # NOTE: all data is ~120GB (2001-2020)

    dataset = "mantle_modis"

    def __init__(self, data_folder: Path = Path("data")) -> None:
        super().__init__(data_folder)

        # boto for accessing AWS S3 buckets
        global boto3
        if boto3 is None:
            import boto3
        global botocore
        global Config
        if botocore is None:
            import botocore
            from botocore.client import Config

        #  gdal to convert tif to netcdf
        global gdal
        if gdal is None:
            from osgeo import gdal

        self.modis_bucket = "mantlelabs-eu-modis-boku"
        self.client = boto3.client(  # type: ignore
            "s3", **DEFAULT_ARGS
        )

    ##########################################
    #  Helper Functions for working with .tif #
    ##########################################

    def get_tif_filepaths(self) -> List[Path]:
        target_folder = self.output_folder
        outfiles = list(target_folder.glob("**/*.tif"))
        outfiles.sort()
        return outfiles

    def delete_tifs_already_run(self):
        dst_dir = self.output_folder / "tifs"
        moved_tif_files = [f for f in dst_dir.glob("*.tif")]
        created_nc_files = [d for d in self.output_folder.glob("**/*.nc")]

        already_converted = np.isin([tif.stem for tif in moved_tif_files], [nc.stem for nc in created_nc_files])
        converted_tifs = np.array(moved_tif_files)[already_converted]

        # delete the already removed
        [f.unlink() for f in converted_tifs]  # type: ignore

    @staticmethod
    def tif_to_nc(tif_file: Path, nc_file: Path, variable: str) -> None:
        """convert .tif -> .nc using GDAL"""
        #  with XARRAY (rasterio backend)
        ds = xr.open_rasterio(tif_file.resolve())
        da = ds.isel(band=0).drop("band")
        da = da.rename({"x": "lon", "y": "lat"})
        da.name = variable
        try:
            da.to_netcdf(nc_file)
        except RuntimeError:
            print("RUN OUT OF MEMORY - deleting the tifs that are already created")
            self.delete_tifs_already_run()

    def preprocess_tif_to_nc(
        self, tif_files: List[Path], variable: str, remove_tif: bool = False
    ) -> None:
        """Create the temporary folder for storing the tif / netcdf files
        (requires copying and deleting).

        NOTE: this function removes the raw tif data from the raw folder.
        """
        # 1. move tif files to /tif directory
        dst_dir = self.output_folder / "tifs"
        if not dst_dir.exists():
            dst_dir.mkdir(exist_ok=True, parents=True)

        dst_tif_files = [dst_dir / f.name for f in tif_files]

        for src, dst in zip(tif_files, dst_tif_files):
            shutil.move(src, dst)

        # 2. convert from tif to netcdf (complete in RAW directory)
        moved_tif_files = [f for f in dst_dir.glob("*.tif")]
        nc_files = [f.parents[0] / (f.stem + ".nc") for f in tif_files]
        moved_tif_files.sort()
        nc_files.sort()

        print("\n")
        for tif_file, nc_file in zip(moved_tif_files, nc_files):
            self.tif_to_nc(tif_file, nc_file, variable=variable)
            print(f"-- Converted {tif_file.name} to netcdf --")

        # 3. remove the tif files
        if remove_tif:
            [f.unlink() for f in moved_tif_files]  # type: ignore

    ##########################################
    #  Helper Functions for working with S3   #
    ##########################################

    def _get_filepaths(self, year: int, month: int) -> List[str]:
        # because of API limits - combine lists by splitting up
        # calls by year / month
        target_prefix = f"modis_boku-preprocess/v1.0/MCD13A2_{year}{month:02d}"

        files = self.client.list_objects_v2(
            Bucket=self.modis_bucket, Prefix=target_prefix
        )
        file_strs = []
        try:
            for file in files["Contents"]:
                key = file["Key"]
                file_strs.append(key)
        except KeyError:
            print(f"No data found for {year}-{month:02d}")
            pass

        return file_strs

    def _list_objects(self, target_prefix: str) -> Any:
        paginator = self.client.get_paginator("list_objects")
        result = paginator.paginate(  # type: ignore
            Bucket=self.modis_bucket, Delimiter="/", Prefix=target_prefix,
        )
        return result

    def _get_all_containing_folders(self,) -> Dict[str, List[str]]:
        """This method allows the exporter to identify how much data is
        currently available in the bucket
        """
        target_prefix = f"modis_boku-preprocess/v1.0/MCD13A2"
        result = self._list_objects(  # type: ignore
            target_prefix=target_prefix
        )

        # save results to dictionary
        index_dict: Dict[str, List[str]] = {
            "O0": [],
            "O1": [],
            "O2": [],
            "O3": [],
            "O4": [],
            "OF": [],
        }
        for prefix in result.search("CommonPrefixes"):
            lvl: str = prefix["Prefix"].split("_")[-1][:-1]
            assert lvl in list(
                index_dict.keys()
            ), f"Expect to find {lvl} in {list(index_dict.keys())}"
            index_dict[lvl].append(prefix["Prefix"])

        return index_dict

    def _get_all_files(
        self,
        years: Optional[List[int]] = None,
        months: Optional[List[int]] = None,
        level: str = "OF",
    ):
        if months is None:
            months = np.arange(1, 13)
        if years is None:
            df = self._get_all_folders(level=level)
            years = df["year"].unique()

        all_files = []
        for year, month in product(years, months):
            # get the list of all the filepaths for that year/month
            all_files.extend(self._get_filepaths(year=year, month=month))

        return all_files

    def _get_all_folders(self, level) -> pd.DataFrame:
        index_dict = self._get_all_containing_folders()

        # get all folders of the given level: {O0, O1, O2, O3, O4, OF}
        df = pd.DataFrame(
            {"folder": index_dict[level]}, index=np.arange(len(index_dict[level]))
        )
        df["time"] = pd.to_datetime(
            [folder.split("_")[2] for folder in index_dict[level]]
        )
        df["year"] = df["time"].dt.year
        df["month"] = df["time"].dt.month

        return df

    ##########################################
    # Export Functions                       #
    ##########################################

    def _export_list_of_files(
        self, subset_files: List[str], verbose: bool = False
    ) -> None:
        # get datetimes of the subset_files
        datetimes = pd.to_datetime([f.split("_")[2] for f in subset_files])

        output_files = []
        # Download each file, creating out folder structure
        # data_dir / raw / mantle_modis / MCD13A2_006_globalV1_1km_OF / {year}
        for target_key, dt in tqdm(zip(subset_files, datetimes)):
            # create filename (target_name)
            path = Path(target_key)
            target_name = f"{dt.year}{dt.month:02d}{dt.day:02d}_{path.name}"

            # create folder structure (target_folder)
            level = target_key.split("/")[2].split("_")[-1]
            variable = target_key.split("/")[-2].lower()
            folder_level = f"{variable}_{level}"
            target_folder = self.output_folder / folder_level / str(dt.year)
            target_folder.mkdir(parents=True, exist_ok=True)

            #  create the output file (target_output)
            target_output = target_folder / target_name

            # check that it doesn't already exist ...
            if target_output.exists():
                print(f"{target_output} already exists! Skipping")
                continue

            try:
                self.client.download_file(
                    Bucket=self.modis_bucket,
                    Key=target_key,
                    Filename=str(target_output),
                )
                output_files.append(target_output)
                if verbose:
                    print(f"Exported {target_key} to {target_folder}")
            except botocore.exceptions.ClientError as e:  # type: ignore
                if e.response["Error"]["Code"] == "404":
                    warnings.warn("Key does not exist! " f"{target_key}")
                else:
                    raise e

    def export(
        self,
        variable: str = "vci",
        level: str = "OF",
        years: Optional[List[int]] = None,
        months: Optional[List[int]] = None,
        remove_tif: bool = False,
    ):
        assert variable in [
            "sm",
            "ndvi",
            "vci",
            "uup",
            "ulow",
        ], f"Expect var: {variable} to be one of [sm, ndvi, vci, uup, ulow]"
        assert level in [
            "O0",
            "O1",
            "O2",
            "O3",
            "O4",
            "OF",
        ], f"Expect level {level} to be one of [O0, O1, O2, O3, O4, OF]"

        # return ALL files (TODO: definitely a speedier way of achieving)
        all_files = self._get_all_files(years=years, months=months, level=level)

        # get the variable and level for each filename
        variables = np.array(
            [f.split("/")[-1].replace(".tif", "").lower() for f in all_files]
        )
        levels = np.array(
            [f.split("1km")[-1].split("/")[0].replace("_", "") for f in all_files]
        )

        # filter by level and variable
        subset = (variables == variable) & (levels == level)
        subset_files = np.array(all_files)[subset]

        self._export_list_of_files(subset_files)

        # convert tif to netcdf
        out_tif_files = self.get_tif_filepaths()

        print("\n** Exported TIFs. Now Processing to NETCDF **\n")
        self.preprocess_tif_to_nc(
            out_tif_files, remove_tif=remove_tif, variable=f"modis_{variable}"
        )
