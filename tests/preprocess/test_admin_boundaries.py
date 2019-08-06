from src.preprocess.admin_boundaries import OCHAAdminBoundariesPreprocesser
from src.preprocess import KenyaAdminPreprocessor
from ..utils import _make_dataset, _create_demo_shapefile


class TestAdminBoundariesPreprocessor:
    @staticmethod
    def test_init(tmp_path):
        preprocessor = OCHAAdminBoundariesPreprocesser(tmp_path)
        assert preprocessor.out_dir.parts[-2] == 'analysis', 'self.analysis should' \
            'be True and the output directory should be analysis'

        assert (tmp_path / 'analysis' / 'boundaries_preprocessed').exists()

    @staticmethod
    def test_existing_file_not_overwritten(tmp_path, capsys):
        ref_nc_dir = tmp_path / 'interim' / 'vhi_preprocessed'
        ref_nc_dir.mkdir(exist_ok=True, parents=True)
        reference_nc_filepath = (ref_nc_dir / 'vhi_kenya.nc')
        reference_nc_filepath.touch()

        shp_dir = tmp_path / 'analysis' / 'boundaries_preprocessed'
        shp_dir.mkdir(exist_ok=True, parents=True)
        shp_filepath = shp_dir / 'province_l1_kenya.nc'
        shp_filepath.touch()

        preprocessor = KenyaAdminPreprocessor(tmp_path)
        preprocessor._preprocess_single(
            shp_filepath=shp_filepath,
            lookup_colname='PROVINCE',
            reference_nc_filepath=reference_nc_filepath,
            var_name='province_l1'
        )

        captured = capsys.readouterr()
        expected_stdout = '** Data already preprocessed!'
        assert expected_stdout in captured.out

    def test_preprocess(self, tmp_path):
        shp_filepath = (
            tmp_path / 'raw' / 'boundaries' / 'kenya' / 'Admin2/KEN_admin2_2002_DEPHA.shp'
        )
        shp_filepath.parents[0].mkdir(parents=True, exist_ok=True)
        _create_demo_shapefile(shp_filepath)

        ref_nc_dir = tmp_path / 'interim' / 'vhi_preprocessed'
        ref_nc_dir.mkdir(exist_ok=True, parents=True)
        reference_nc_filepath = ref_nc_dir / 'vhi_kenya.nc'
        reference_ds, _, _ = _make_dataset((30, 30))
        reference_ds.to_netcdf(reference_nc_filepath)

        preprocessor = KenyaAdminPreprocessor(tmp_path)
        preprocessor.preprocess(
            reference_nc_filepath=reference_nc_filepath, selection='level_1'
        )

        assert False
