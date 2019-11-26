import pytest
import xarray as xr

from src.preprocess.admin_boundaries import OCHAAdminBoundariesPreprocesser
from src.preprocess import KenyaAdminPreprocessor
from ..utils import _make_dataset, CreateSHPFile


class TestAdminBoundariesPreprocessor:
    @staticmethod
    @pytest.mark.xfail(reason="geopandas not part of the testing environment")
    def test_init(tmp_path):
        preprocessor = OCHAAdminBoundariesPreprocesser(tmp_path)
        assert preprocessor.out_dir.parts[-2] == "analysis", (
            "self.analysis should" "be True and the output directory should be analysis"
        )

        assert (tmp_path / "analysis" / "boundaries_preprocessed").exists()

    @staticmethod
    @pytest.mark.xfail(reason="geopandas not part of the testing environment")
    def test_existing_file_not_overwritten(tmp_path, capsys):
        ref_nc_dir = tmp_path / "interim" / "vhi_preprocessed"
        ref_nc_dir.mkdir(exist_ok=True, parents=True)
        reference_nc_filepath = ref_nc_dir / "vhi_kenya.nc"
        reference_nc_filepath.touch()

        shp_dir = tmp_path / "analysis" / "boundaries_preprocessed"
        shp_dir.mkdir(exist_ok=True, parents=True)
        shp_filepath = shp_dir / "province_l1_kenya.nc"
        shp_filepath.touch()

        preprocessor = KenyaAdminPreprocessor(tmp_path)
        preprocessor._preprocess_single(
            shp_filepath=shp_filepath,
            lookup_colname="PROVINCE",
            reference_nc_filepath=reference_nc_filepath,
            var_name="province_l1",
        )

        captured = capsys.readouterr()
        expected_stdout = "** Data already preprocessed!"
        assert expected_stdout in captured.out

    @pytest.mark.xfail(reason="geopandas not part of the testing environment")
    def test_preprocess(self, tmp_path):
        shp_filepath = (
            tmp_path
            / "raw"
            / "boundaries"
            / "kenya"
            / "Admin2/KEN_admin2_2002_DEPHA.shp"
        )
        shp_filepath.parents[0].mkdir(parents=True, exist_ok=True)
        s = CreateSHPFile()
        s.create_demo_shapefile(shp_filepath)

        ref_nc_dir = tmp_path / "interim" / "vhi_preprocessed"
        ref_nc_dir.mkdir(exist_ok=True, parents=True)
        reference_nc_filepath = ref_nc_dir / "vhi_kenya.nc"
        reference_ds, _, _ = _make_dataset(
            size=(30, 30), latmin=-1, latmax=1, lonmin=33, lonmax=35
        )
        reference_ds.to_netcdf(reference_nc_filepath)

        preprocessor = KenyaAdminPreprocessor(tmp_path)
        preprocessor.preprocess(
            reference_nc_filepath=reference_nc_filepath, selection="level_1"
        )

        assert (preprocessor.out_dir / "province_l1_kenya.nc").exists()

        # read the resulting xarray object
        ds = xr.open_dataset((preprocessor.out_dir / "province_l1_kenya.nc"))
        assert "province_l1" == [d for d in ds.data_vars][0]
        assert ["lat", "lon"] == [d for d in ds.coords]

        # extract the dictionary lookup from the attrs
        lookup_dict = dict(
            zip(ds.attrs["keys"].split(", "), ds.attrs["values"].split(", "))
        )
        assert lookup_dict["0"] == "NAIROBI"
