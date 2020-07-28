import xarray as xr
import numpy as np

from src.preprocess.boku_ndvi import BokuNDVIPreprocessor

from src.utils import get_kenya, get_ethiopia

from ..utils import _make_dataset

"""
TODO:
- Test both 250m and 1000m
- Test values from int scale -> ndvi float scale
- Test the removal of 251, 252, 255 values
"""


class TestBokuNDVIPreprocessor:
    @staticmethod
    def test_make_filename():

        test_file = "testy_test.nc"
        expected_output = "testy_test_kenya.nc"

        filename = BokuNDVIPreprocessor.create_filename(test_file, "kenya")
        assert (
            filename == expected_output
        ), f"Expected output to be {expected_output}, got {filename}"

    @staticmethod
    def _make_boku_ndvi_dataset(
        size,
        lonmin=-180.0,
        lonmax=180.0,
        latmin=-55.152,
        latmax=75.024,
        kenya_only=False,
    ):
        lat_len, lon_len = size
        if kenya_only:
            kenya = get_kenya()
            latmin = kenya.latmin
            latmax = kenya.latmax
            lonmin = kenya.lonmin
            lonmax = kenya.lonmax

        # create the vector
        longitudes = np.linspace(lonmin, lonmax, lon_len)
        latitudes = np.linspace(latmin, latmax, lat_len)

        dims = ["lat", "lon"]
        coords = {"lat": latitudes, "lon": longitudes}

        modis_vals = np.append(np.arange(1, 252), 255)
        data = np.random.choice(modis_vals, size=size)

        return xr.Dataset({"boku_ndvi": (dims, data)}, coords=coords)

    @staticmethod
    def test_directories_created(tmp_path):
        v = BokuNDVIPreprocessor(tmp_path)

        assert (
            v.preprocessed_folder / "boku_ndvi_1000_preprocessed"
        ).exists(), "Should have created a directory tmp_path/interim/boku_ndvi_1000_preprocessed"

        assert (
            v.preprocessed_folder / "boku_ndvi_1000_interim"
        ).exists(), (
            "Should have created a directory tmp_path/interim/boku_ndvi_1000_interim"
        )

    @staticmethod
    def test_get_filenames(tmp_path):

        (tmp_path / "raw" / "boku_ndvi_1000").mkdir(parents=True)

        test_file = tmp_path / "raw/boku_ndvi_1000/testy_test.nc"
        test_file.touch()

        processor = BokuNDVIPreprocessor(tmp_path)

        files = processor.get_filepaths()
        assert files[0] == test_file, f"Expected {test_file} to be retrieved"

    def test_preprocess(self, tmp_path):

        (tmp_path / "raw/boku_ndvi_1000").mkdir(parents=True)

        RAW_FILES = [
            "MCD13A2.t200915.006.EAv1.1_km_10_days_NDVI.O1.nc",
            "MCD13A2.t201107.006.EAv1.1_km_10_days_NDVI.O1.nc",
            "MCD13A2.t201330.006.EAv1.1_km_10_days_NDVI.O1.nc",
            "MCD13A2.t201733.006.EAv1.1_km_10_days_NDVI.O1.nc",
        ]

        for raw_file in RAW_FILES:
            data_path = tmp_path / f"raw/boku_ndvi_1000/{raw_file}"
            dataset = self._make_boku_ndvi_dataset(size=(100, 100), kenya_only=True)
            dataset.to_netcdf(path=data_path)

        kenya = get_kenya()
        regrid_dataset, _, _ = _make_dataset(
            size=(20, 20),
            latmin=kenya.latmin,
            latmax=kenya.latmax,
            lonmin=kenya.lonmin,
            lonmax=kenya.lonmax,
        )

        regrid_path = tmp_path / "regridder.nc"
        regrid_dataset.to_netcdf(regrid_path)

        processor = BokuNDVIPreprocessor(tmp_path)
        processor.preprocess(subset_str="kenya", regrid=regrid_path, cleanup=True)

        expected_out_path = (
            tmp_path / "interim/boku_ndvi_1000_preprocessed/data_kenya.nc"
        )
        assert (
            expected_out_path.exists()
        ), f"Expected processed file to be saved to {expected_out_path}"

        # check the subsetting happened correctly
        out_data = xr.open_dataset(expected_out_path)
        expected_dims = ["lat", "lon", "time"]
        assert len(list(out_data.dims)) == len(expected_dims)
        for dim in expected_dims:
            assert dim in list(
                out_data.dims
            ), f"Expected {dim} to be in the processed dataset dims"

        lons = out_data.lon.values
        assert (lons.min() >= kenya.lonmin) and (
            lons.max() <= kenya.lonmax
        ), "Longitudes not correctly subset"

        lats = out_data.lat.values
        assert (lats.min() >= kenya.latmin) and (
            lats.max() <= kenya.latmax
        ), "Latitudes not correctly subset"

        assert out_data["boku_ndvi"].values.shape[1:] == (20, 20)

        assert (
            not processor.interim.exists()
        ), f"Interim boku_ndvi folder should have been deleted"

    def test_alternative_region(self, tmp_path):
        # make the dataset
        (tmp_path / "raw/boku_ndvi_1000").mkdir(parents=True)
        data_path = (
            tmp_path
            / "raw/boku_ndvi_1000/MCD13A2.t200915.006.EAv1.1_km_10_days_NDVI.O1.nc"
        )
        dataset = self._make_boku_ndvi_dataset(size=(100, 100))
        dataset.to_netcdf(path=data_path)
        ethiopia = get_ethiopia()

        # regrid the datasets
        regrid_dataset, _, _ = _make_dataset(
            size=(20, 20),
            latmin=ethiopia.latmin,
            latmax=ethiopia.latmax,
            lonmin=ethiopia.lonmin,
            lonmax=ethiopia.lonmax,
        )
        regrid_path = tmp_path / "regridder.nc"
        regrid_dataset.to_netcdf(regrid_path)

        # build the Preprocessor object and subset with a different subset_str
        processor = BokuNDVIPreprocessor(tmp_path)
        processor.preprocess(subset_str="ethiopia", regrid=regrid_path, n_processes=1)

        expected_out_path = (
            tmp_path / "interim/boku_ndvi_1000_preprocessed/data_ethiopia.nc"
        )
        assert (
            expected_out_path.exists()
        ), f"Expected processed file to be saved to {expected_out_path}"
