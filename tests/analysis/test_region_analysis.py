from src.analysis import RegionAnalysis
from ..utils import _make_dataset


class TestRegionAnalysis:
    @staticmethod
    def _create_dummy_true_preds_data(tmp_path):
        # save the preds
        parent_dir = tmp_path / 'models' / 'one_month_forecast'
        parent_dir.mkdir(exist_ok=True, parents=True)
        save_fnames = ['2018_1.nc', '2018_2.nc', '2018_3.nc']
        times = ['2018-01-31', '2018-02-28', '2018-03-31']
        for fname, time in zip(save_fnames, times):
            ds = _make_dataset((30, 30), variable_name='VHI',
                               lonmin=30, lonmax=35,
                               latmin=-2, latmax=2,
                               start_date=time, end_date=time)
            ds.to_netcdf(parent_dir / fname)

        # save the TRUTH (test files)
        save_dnames = ['2018_1', '2018_2', '2018_3']
        parent_dir = tmp_path / 'features' / 'test' / 'one_month_forecast'
        parent_dir.mkdir(exist_ok=True, parents=True)
        for dname, time in zip(save_dnames, times):
            ds = _make_dataset((30, 30), variable_name='VHI',
                               lonmin=30, lonmax=35,
                               latmin=-2, latmax=2,
                               start_date=time, end_date=time)

            (parent_dir / dname).mkdir(exist_ok=True, parents=True)
            ds.to_netcdf(parent_dir / dname / 'y.nc')

    @staticmethod
    def _create_dummy_admin_boundaries_data(tmp_path):
        ds = _make_dataset((30, 30), variable_name='VHI',
                   lonmin=30, lonmax=35,
                   latmin=-2, latmax=2,
                   start_date=time, end_date=time)
        ds.VHI.astype(int)

        (tmp_path / 'analysis' / 'boundaries_preprocessed').mkdir(
            exist_ok=True, parents=True
        )
        ds.to_netcdf(tmp_path / 'analysis' / 'boundaries_preprocessed' / 'province_l1_kenya.nc')

    @staticmethod
    def test_init(tmp_path):
        analyser = RegionAnalysis(tmp_path, admin_boundaries=True)

        assert (tmp_path / 'analysis' / 'region_analysis').exists()
        assert analyser.shape_data_dir.name == 'boundaries_preprocessed'

    def test_analyser(self, tmp_path):
        self._create_dummy_true_preds_data(tmp_path)
        self._create_dummy_admin_boundaries_data(tmp_path)

        analyser = RegionAnalysis(tmp_path, admin_boundaries=True)

        assert len(analyser.region_data_paths) == 3, 'should have found the '\
            '3 dummy regional_shapes'
