import numpy as np
from pathlib import Path
import xarray as xr

from typing import cast, Dict, List, Union, Tuple


class Engineer:

    def __init__(self, data_folder: Path = Path('data')) -> None:

        self.data_folder = data_folder

        self.interim_folder = data_folder / 'interim'
        assert self.interim_folder.exists(), \
            f'{data_folder / "interim"} does not exist. Has the preprocesser been run?'

        self.output_folder = data_folder / 'features'
        if not self.output_folder.exists():
            self.output_folder.mkdir()

    def engineer(self, test_year: Union[int, List[int]],
                 target_variable: str = 'VHI'):
        data = self._make_dataset()

        if type(test_year) is int:
            test_year = [cast(int, test_year)]

        train, test = self._train_test_split(data, cast(List, test_year), target_variable)
        self._save(train, test)

    def _get_preprocessed_files(self) -> List[Path]:
        processed_files = []
        for subfolder in self.interim_folder.iterdir():
            if str(subfolder).endswith('_preprocessed') and subfolder.is_dir():
                processed_files.extend(list(subfolder.glob('*.nc')))
        return processed_files

    def _make_dataset(self) -> xr.Dataset:

        datasets = []
        dims = ['lon', 'lat']
        coords = {}
        for idx, file in enumerate(self._get_preprocessed_files()):
            datasets.append(xr.open_dataset(file))

            if idx == 0:
                for dim in dims:
                    coords[dim] = datasets[idx][dim].values
            else:
                for dim in dims:
                    assert (datasets[idx][dim].values == coords[dim]).all(), \
                        f'{dim} is different! Was this run using the preprocessor?'

        main_dataset = datasets[0]
        for dataset in datasets[1:]:
            main_dataset = main_dataset.merge(dataset, join='inner')

        return main_dataset

    def _train_test_split(self, ds: xr.Dataset,
                          years: List[int],
                          target_variable: str = 'VHI'
                          ) -> Tuple[xr.Dataset, Dict[int, xr.Dataset]]:
        years.sort()

        output_test_arrays = {}

        train, test = self._train_test_split_single_year(ds, years[0],
                                                         target_variable)
        output_test_arrays[years[0]] = test

        if len(years) > 1:
            for year in years[1:]:
                _, subtest = self._train_test_split_single_year(ds, year,
                                                                target_variable)
                output_test_arrays[year] = subtest

        return train, output_test_arrays

    @staticmethod
    def _train_test_split_single_year(ds: xr.Dataset,
                                      year: int,
                                      target_variable: str
                                      ) -> Tuple[xr.Dataset, xr.Dataset]:

        min_date = np.datetime64(f'{year}-01-01')
        max_date = np.datetime64(f'{year}-12-31')

        train = ds.time.values < min_date
        test = ((ds.time.values >= min_date) & (ds.time.values <= max_date))

        test_dataset = ds.isel(time=test)[target_variable].to_dataset()

        return ds.isel(time=train), test_dataset

    def _save(self, train: xr.Dataset, test: Dict[int, xr.Dataset]):
        train.to_netcdf(self.output_folder / 'train.nc')

        for key, val in test.items():

            filename = f'test_{key}.nc'

            val.to_netcdf(self.output_folder / filename)
