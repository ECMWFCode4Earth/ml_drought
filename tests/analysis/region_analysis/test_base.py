import numpy as np
import xarray as xr

from src.analysis.region_analysis.base import RegionAnalysis
from tests.utils import _make_dataset


class TestRegionAnalysis:
    @staticmethod
    def _create_dummy_landcover_data(tmp_path):
        parent_dir = tmp_path / "interim" / "static" / "esa_cci_landcover_preprocessed"
        parent_dir.mkdir(exist_ok=True, parents=True)
        fname = "esa_cci_landcover_kenya_one_hot.nc"
        vars = [
            "Cropland, irrigated or post-flooding_one_hot",
            "Herbaceous cover_one_hot",
            "No data_one_hot",
            "Tree or shrub cover_one_hot",
        ]
        # create non-overlapping groups
        # https://stackoverflow.com/a/52356978/9940782
        groups = np.random.randint(0, 4, (30, 30))
        masks = (groups[..., None] == np.arange(4)[None, :]).T.astype(int)

        all_ds = []
        for group, var in enumerate(vars):
            ds, _, _ = _make_dataset(
                (30, 30),
                variable_name=var,
                lonmin=30,
                lonmax=35,
                latmin=-2,
                latmax=2,
                add_times=False,
                const=True,
            )
            # assign the values from the mask to the da.values
            ds[var].values = masks[group, :, :]
            all_ds.append(ds)

        ds = xr.merge([*all_ds])
        ds.to_netcdf(parent_dir / fname)

    @staticmethod
    def _create_dummy_true_preds_data(tmp_path):
        # save the preds
        parent_dir = tmp_path / "models" / "one_month_forecast" / "ealstm"
        parent_dir.mkdir(exist_ok=True, parents=True)
        save_fnames = ["preds_2018_1.nc", "preds_2018_2.nc", "preds_2018_3.nc"]
        times = ["2018-01-31", "2018-02-28", "2018-03-31"]
        for fname, time in zip(save_fnames, times):
            ds, _, _ = _make_dataset(
                (30, 30),
                variable_name="VHI",
                lonmin=30,
                lonmax=35,
                latmin=-2,
                latmax=2,
                start_date=time,
                end_date=time,
            )
            ds.to_netcdf(parent_dir / fname)

        # save the TRUTH (test files)
        save_dnames = ["2018_1", "2018_2", "2018_3"]
        parent_dir = tmp_path / "features" / "one_month_forecast" / "test"
        parent_dir.mkdir(exist_ok=True, parents=True)
        for dname, time in zip(save_dnames, times):
            ds, _, _ = _make_dataset(
                (30, 30),
                variable_name="VHI",
                lonmin=30,
                lonmax=35,
                latmin=-2,
                latmax=2,
                start_date=time,
                end_date=time,
            )

            (parent_dir / dname).mkdir(exist_ok=True, parents=True)
            ds.to_netcdf(parent_dir / dname / "y.nc")

    @staticmethod
    def _create_dummy_admin_boundaries_data(tmp_path, prefix: str):
        ds, _, _ = _make_dataset(
            (30, 30),
            variable_name="VHI",
            lonmin=30,
            lonmax=35,
            latmin=-2,
            latmax=2,
            add_times=False,
        )
        ds.VHI.astype(int)

        (tmp_path / "analysis" / "boundaries_preprocessed").mkdir(
            exist_ok=True, parents=True
        )
        ds.attrs["keys"] = ", ".join([str(i) for i in range(3)])
        ds.attrs["values"] = ", ".join([f"region_{i}" for i in np.arange(0, 3)])
        ds.to_netcdf(
            tmp_path
            / "analysis"
            / "boundaries_preprocessed"
            / f"province_l{prefix}_kenya.nc"
        )

    def test_init(self, tmp_path):
        self._create_dummy_true_preds_data(tmp_path)
        self._create_dummy_admin_boundaries_data(tmp_path, prefix="")
        analyser = RegionAnalysis(data_dir=tmp_path)

        assert (tmp_path / "analysis" / "region_analysis").exists()
        assert analyser.shape_data_dir.name == "boundaries_preprocessed"
