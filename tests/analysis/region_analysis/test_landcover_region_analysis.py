import numpy as np
import pandas as pd
import xarray as xr

from src.analysis import LandcoverRegionAnalysis
from tests.utils import _make_dataset
from .test_region_analysis import TestRegionAnalysis


class TestLandcoverRegionAnalysis(TestRegionAnalysis):
    @staticmethod
    def _create_dummy_landcover_data(tmp_path):
        parent_dir = tmp_path / 'interim' / 'static' / 'esa_cci_landcover_preprocessed'
        parent_dir.mkdir(exist_ok=True, parents=True)
        fname = 'esa_cci_landcover_kenya_one_hot.nc'
        vars = [
            'Cropland, irrigated or post-flooding_one_hot',
            'Herbaceous cover_one_hot',
            'No data_one_hot',
            'Tree or shrub cover_one_hot'
        ]
        # create non-overlapping groups
        # https://stackoverflow.com/a/52356978/9940782
        groups = np.random.randint(0, 4, (30, 30))
        masks = (groups[..., None] == np.arange(4)[None, :]).T.astype(int)

        all_ds = []
        for group, var in enumerate(vars):
            ds, _, _ = _make_dataset((30, 30), variable_name=var,
                                     lonmin=30, lonmax=35,
                                     latmin=-2, latmax=2,
                                     add_times=False, const=True)
            # assign the values from the mask to the da.values
            ds[var].values = masks[group, :, :]
            all_ds.append(ds)

        ds = xr.merge([*all_ds])
        ds.to_netcdf(parent_dir / fname)

    def test_analyzer_analyze(self, tmp_path):
        self._create_dummy_landcover_data(tmp_path)
        self._create_dummy_true_preds_data(tmp_path)

        # admin_boundaries=False for landcover grouping
        analyser = LandcoverRegionAnalysis(tmp_path)
        assert 'landcover_kenya_one_hot' in analyser.region_data_paths[0].name

        lcover_das = analyser.load_landcover_data(analyser.region_data_paths[0])
        assert isinstance(lcover_das, list)

        analyser._analyze_single(analyser.region_data_paths[0])

        csv_path = tmp_path / 'analysis' / 'region_analysis' / 'ealstm' / 'ealstm_landcover.csv'
        assert (csv_path).exists()

        df = pd.read_csv(csv_path)
        assert df.admin_level_name.unique() == ['landcover']

        valid_landcover_names = [
            'cropland_irrigated_or_postflooding_one_hot',
            'herbaceous_cover_one_hot',
            'no_data_one_hot',
            'tree_or_shrub_cover_one_hot',
        ]
        assert np.isin(valid_landcover_names, df.region_name.unique()).all()

        n_datetimes = 3
        n_lc_regions = len(valid_landcover_names)
        assert len(df) == (n_datetimes * n_lc_regions * len(['landcover']))
