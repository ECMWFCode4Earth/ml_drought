from pathlib import Path
import xarray as xr

from typing import List, Union


class Engineer:

    def __init__(self, data_folder: Path = Path('data')) -> None:

        self.data_folder = data_folder

        self.interim_folder = data_folder / 'interim'
        assert self.interim_folder.exists(), \
            f'{data_folder / "interim"} does not exist. Has the preprocesser been run?'

        self.output_folder = data_folder / 'features'
        if not self.output_folder.exists():
            self.output_folder.mkdir()

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

    def engineer(self, test_year: Union[int, List[int]],
                 target_variable: str = 'VHI'):

        raise NotImplementedError
