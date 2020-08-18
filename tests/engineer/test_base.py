from src.engineer.base import _EngineerBase as Engineer

from ..utils import _make_dataset


def _setup(data_path, add_times=True, static=False):
    # setup
    interim_folder = data_path / "interim"
    interim_folder.mkdir(exist_ok=True)

    if static:
        interim_folder = interim_folder / "static"
        interim_folder.mkdir(exist_ok=True)

    expected_output, expected_vars = [], []
    for var in ["a", "b"]:
        (interim_folder / f"{var}_preprocessed").mkdir(exist_ok=True, parents=True)

        # this file should be captured
        data, _, _ = _make_dataset((10, 10), var, const=True, add_times=add_times)
        filename = interim_folder / f"{var}_preprocessed/hello.nc"
        data.to_netcdf(filename)

        expected_output.append(filename)
        expected_vars.append(var)

        # this file should not
        (interim_folder / f"{var}_preprocessed/boo").touch()

    # none of this should be captured
    (interim_folder / "woops").mkdir()
    woops_data, _, _ = _make_dataset((10, 10), "oops")
    woops_data.to_netcdf(interim_folder / "woops/hi.nc")

    return expected_output, expected_vars


class TestEngineer:
    def test_get_preprocessed(self, tmp_path, monkeypatch):

        expected_files, expected_vars = _setup(tmp_path)

        def mock_init(self, data_folder):
            self.name = "dummy"
            self.interim_folder = data_folder / "interim"

        monkeypatch.setattr(Engineer, "__init__", mock_init)

        engineer = Engineer(tmp_path)
        files = engineer._get_preprocessed_files(static=False)

        assert set(expected_files) == set(files), f"Did not retrieve expected files!"

    def test_join(self, tmp_path, monkeypatch):

        expected_files, expected_vars = _setup(tmp_path)

        def mock_init(self, data_folder):
            self.name = "dummy"
            self.interim_folder = data_folder / "interim"

        monkeypatch.setattr(Engineer, "__init__", mock_init)

        engineer = Engineer(tmp_path)
        joined_ds = engineer._make_dataset(static=False)

        dims = ["lon", "lat", "time"]
        output_vars = [var for var in joined_ds.variables if var not in dims]

        assert set(output_vars) == set(
            expected_vars
        ), f"Did not retrieve all the expected variables!"
