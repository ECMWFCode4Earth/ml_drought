from pathlib import Path
from unittest.mock import patch
import pytest
import re

from src.exporters.vhi import VHIExporter, _parse_time_from_filename, make_filename


class TestVHIExporter:
    def test_api_locally(self):
        if "tommylees" in Path(".").absolute().as_posix():
            if True:
                return
                assert True, f"Switch to run this function (takes time)"
            fnames = VHIExporter.get_ftp_filenames(years=list(range(2000, 2020)))

            weeks = [int(_parse_time_from_filename(f)[-1]) for f in fnames]
            years = [int(_parse_time_from_filename(f)[0]) for f in fnames]

            # check that our minimum years and max years are correct
            assert (min(weeks) == 1) and (
                max(weeks) == 52
            ), f"Week numbers should \
            be between 1 and 52"
            assert (min(years) >= 1981) and (
                max(years) <= 2019
            ), f"Year numbers \
            should be between 1981 and greater than / equal to 2019"

            # assert that only getting the VH.nc files (not *ND.nc / *SM.nc)
            date = re.compile(r"\d{7}.")
            assert all(
                ["VH" in date.split(f)[-1] for f in fnames]
            ), f"We only \
                want to download the finished VH products not the ND / SM products"

            # assert that correct year
            years.sort()
            assert (
                years[0] == 2000
            ), f"Expected the minimum year to be 2000, Got: {years[0]}"

        else:
            assert (
                True
            ), "This class is only run locally to avoid externally \
                calling ftp servers (on travis for example)"

    def test_filename_illegitimate(self, tmp_path):
        with pytest.raises(Exception) as e:
            make_filename(
                raw_folder=tmp_path / "data",
                raw_filename="this/should/fail.nc",
                dataset="vhi",
            )
            e.match(r"filename cannot have subdirectories*")

    def test_parse_time_from_filename(self):
        fnames = ["VHP.G04.C07.NC.P1981035.VH.nc"]

        weeks = [int(_parse_time_from_filename(f)[-1]) for f in fnames]
        years = [int(_parse_time_from_filename(f)[0]) for f in fnames]

        # check that our minimum years and max years are correct
        assert weeks[0] == 35, f"weeks Expected: [35] Got: {weeks}"
        assert years[0] == 1981, f"years Expected: [1981] Got: {years}"

    @patch("ftplib.FTP", autospec=True)
    def test_ftp_called(self, mock_ftp):
        _ = VHIExporter.get_ftp_filenames(years=list(range(1981, 2020)))
        mock_ftp.assert_called_once_with("ftp.star.nesdis.noaa.gov")

    def test_dir_structure_create(self, tmp_path):
        VHIExporter(tmp_path)
        raw_folder = tmp_path / "raw"
        raw_filename = "VHP.G04.C07.NC.P1981035.VH.nc"

        expected_path = raw_folder / "vhi/1981/VHP.G04.C07.NC.P1981035.VH.nc"
        filename = make_filename(raw_folder, raw_filename, "vhi")

        assert expected_path == filename, f"Got {filename}, expected {expected_path}"
        assert raw_folder.exists(), f"`raw` directory should exist"
        assert (
            raw_folder / "vhi" / "1981"
        ).exists(), f"`raw/vhi/1981` directory should exist"
