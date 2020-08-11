from pathlib import Path
from shutil import rmtree
import xarray as xr
import os
import subprocess

from typing import Optional

from .base import BasePreProcessor


class SRTMPreprocessor(BasePreProcessor):
    dataset = "srtm"
    static = True

    def regrid(  # type: ignore
        self, ds: xr.Dataset, regrid: Path, method: str = "remapbil"
    ) -> xr.Dataset:
        """regrid using CDO (because these SRTM files tend to be large).

        Args:
            ds (xr.Dataset): dataset to be regrid
            regrid (Path): the reference dataset to use as the target grid
            method (str, optional): the method of regridding. Defaults to "remapbil".

        Returns:
            xr.Dataset: the regrid datset (on the target grid)
        """

        acceptable_methods = {
            "remapbil",
            "remapbic",
            "remapnn",
            "remapdis",
            "remapycon",
            "remapcon",
            "remapcon2",
            "remaplaf",
        }
        assert method in acceptable_methods, (
            f"{method} not in {acceptable_methods}, see the interpolation section of "
            f"https://code.mpimet.mpg.de/projects/cdo/wiki/Tutorial for more information. "
            "See https://gist.github.com/mainvoid007/e5f1c82f50eb0459a55dfc4a0953a08e for installation."
        )

        regrid_input = self.interim / "temp.nc"
        regrid_output = self.interim / "temp_regridded.nc"
        input_reference_grid = regrid.resolve().as_posix()
        output_reference_grid = (self.interim / "grid_definition").resolve().as_posix()

        ds.to_netcdf(regrid_input)

        # make the grid definition
        # TODO: change to subprocess.Popen
        # os.system(f"cdo griddes {input_reference_grid} > {output_reference_grid}")
        try:
            retcode = subprocess.call(
                f"cdo griddes {input_reference_grid} > {output_reference_grid}",
                shell=True,
            )
            if retcode != 0:  #  0 means success
                assert False, (
                    f"ERROR: retcode = {retcode}. "
                    "CDO is likely not installed. "
                    "Run: ml_drought/scripts/drafts/install_cdo2.sh "
                    "From: https://gist.github.com/mainvoid007/e5f1c82f50eb0459a55dfc4a0953a08e"
                )
        except OSError as e:
            print(f"Execution failed: {e}")

        # use the grid definition to regrid
        regrid_input_str = regrid_input.resolve().as_posix()
        regrid_output_str = regrid_output.resolve().as_posix()
        # TODO: change to subprocess.Popen
        # os.system(
        #     f"cdo {method},{output_reference_grid} {regrid_input_str} {regrid_output_str}"
        # )
        try:
            retcode = subprocess.call(
                f"cdo {method},{output_reference_grid} {regrid_input_str} {regrid_output_str}",
                shell=True,
            )
            if retcode != 0:  #  0 means success
                assert False, (
                    f"ERROR: retcode = {retcode}. "
                    "CDO is likely not installed. "
                    "Run: ml_drought/scripts/drafts/install_cdo2.sh "
                    "From: https://gist.github.com/mainvoid007/e5f1c82f50eb0459a55dfc4a0953a08e"
                )
        except OSError as e:
            print(f"Execution failed: {e}")

        remapped_ds = xr.open_dataset(regrid_output)

        reference_grid = self.load_reference_grid(regrid)

        # the CDO method yields rounding errors which make merging datasets tricky
        # (e.g. 32.4999 != 32.5). This makes sure the longitude and latitude grids
        # exactly match the reference
        remapped_ds["lon"] = reference_grid.lon
        remapped_ds["lat"] = reference_grid.lat

        return remapped_ds

    def preprocess(
        self,
        subset_str: str = "kenya",
        regrid: Optional[Path] = None,
        cleanup: bool = True,
    ) -> None:
        """Preprocess a downloaded topography .nc file to produce
        one subset file with no timestep

        Arguments:
        ---------
        subset_str: str = 'kenya'
            Because the SRTM data can only be downloaded in tiles, the subsetting happens
            during the export step. This tells the preprocessor which file to preprocess
        regrid: Optional[Path] = None
            If a Path is passed, the CHIRPS files will be regridded to have the same
            grid as the dataset at that Path. If None, no regridding happens
        cleanup: bool = True
            If true, delete interim files created by the class
        """
        print(f"Reading data from {self.raw_folder}. Writing to {self.interim}")

        netcdf_filepath = self.raw_folder / f"{self.dataset}/{subset_str}.nc"

        print(f"Starting work on {netcdf_filepath.name}")
        ds = (
            xr.open_dataset(netcdf_filepath).drop("crs").rename({"Band1": "topography"})
        )

        if regrid is not None:
            print(
                "The SRTM preprocessor requires CDO to be installed! "
                "\nSee here for installation details: "
                "https://code.mpimet.mpg.de/projects/cdo/embedded/index.html#x1-30001.1"
                "\nAnd here for details on using the project: "
                "https://code.mpimet.mpg.de/projects/cdo/wiki/Tutorial"
            )
            ds = self.regrid(ds, regrid)

        print(f"Saving to {self.out_dir}/{subset_str}.nc")
        ds.to_netcdf(str(self.out_dir / f"{subset_str}.nc"))

        print(f"Processed {netcdf_filepath}")

        if cleanup:
            rmtree(self.interim)
