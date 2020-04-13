import xarray as xr
import numpy as np

from src.engineer import DynamicEngineer


class TestDynamicEngineer:
    def test_all(self, tmp_path):
        # make dummy data for the engineer

        # initialise the engineer
        de = DynamicEngineer(tmp_path, process_static=True)
        de.engineer(
            augment_static=False,
            static_ignore_vars=static_ignore_vars,
            dynamic_ignore_vars=dynamic_ignore_vars,
            logy=True,
        )

        # pseudo tests!
        assert de.output_folder.exists()
        assert de.static_output_folder.exists()
        assert all(
            np.isin(
                ["data.nc", "normalizing_dict.pkl"],
                [d.name for d in de.output_folder.iterdir()],
            )
        )
        assert all(
            np.isin(
                ["data.nc", "normalizing_dict.pkl"],
                [d.name for d in de.static_output_folder.iterdir()],
            )
        )
        test_dynamic = xr.open_dataset([d for d in de.output_folder.glob("*.nc")][0])
        test_static = xr.open_dataset(
            [d for d in de.static_output_folder.glob("*.nc")][0]
        )

        assert all(
            np.isin(
                ["peti", "precipitation", "discharge_spec", "target_var_original"],
                list(test_dynamic.data_vars),
            )
        )

        assert all(
            np.isin(
                ["dpsbar", "aridity", "frac_snow", "crop_perc"],
                list(test_static.data_vars),
            )
        )
