from pathlib import Path

from typing import List


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
