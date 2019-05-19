"""

- Add lat lon coordinates
- add time coordinates
- subset Kenya
- merge into one time (~500MB)
"""
import xarray as xr

from .base import (BasePreProcessor,)
from .preprocess_vhi import (
    extract_timestamp,
    create_lat_lon_vectors,
    create_new_dataset,
    create_filename,
)
from .preprocess_utils import select_bounding_box_xarray

from xarray import Dataset


class VHIPreprocesser(BasePreProcessor):
    """ Preprocesses the VHI data """

    def __init__():
        pass

    def read_and_join_nc_files() -> Dataset:
        pass

    def preprocess_VHI_data(self,
                            netcdf_filepath: str,
                            output_dir: str) -> None:
        """Run the Preprocessing steps for the NOAA VHI data

        Process:
        -------
        * assign time stamp
        * assign lat lon
        * create new dataset with these dimensions
        * Save the output file to new folder
        """
        print(f"** Starting work on {netcdf_filepath.split('/')[-1]} **")
        # 1. read in the dataset
        ds = xr.open_dataset(netcdf_filepath)

        # 2. extract the timestamp for that file (from the filepath)
        timestamp = extract_timestamp(ds, netcdf_filepath, use_filepath=True)

        # 3. extract the lat/lon vectors
        longitudes, latitudes = create_lat_lon_vectors(ds)

        # 4. create new dataset with these dimensions
        new_ds = create_new_dataset(ds, longitudes, latitudes, timestamp)

        # 5. chop out EastAfrica
        kenya_region = self.get_kenya()
        kenya_ds = select_bounding_box_xarray(new_ds, kenya_region)

        # 6. create the filepath and save to that location
        filename = create_filename(timestamp, netcdf_filepath, subset=True, subset_name="kenya")
        print(f"Saving to {output_dir}/{filename}")
        # TODO: change to pathlib.Path objects
        kenya_ds.to_netcdf(f"{output_dir}/{filename}")

        print(f"** Done for VHI {netcdf_filepath.split('/')[-1]} **")
