from pathlib import Path
from unittest.mock import patch
import pytest

from src.exporters.vhi import (
    VHIExporter,
    _parse_time_from_filename,
    make_filename,
    # download_file_from_ftp
)

# get the root project directory
project_dir = Path(__file__).resolve().parents[2]


class TestVHIExporter:

    def test_initialisation(self):
        VHIExporter()
        return

    def test_filename_illegitimate(self):
        with pytest.raises(Exception):
            make_filename(
                raw_folder=project_dir / "data",
                raw_filename='this/should/fail.nc',
                dataset='vhi',
            )

        return

    @patch('ftplib.FTP', autospec=True)
    def test_filenames_found(self, mock_ftp_constructor):
        mock_ftp = mock_ftp_constructor.return_value
        vhi = VHIExporter()
        fnames = vhi.get_ftp_filenames()

        mock_ftp_constructor.assert_called_with('ftp.star.nesdis.noaa.gov')
        self.assertTrue(mock_ftp.login.called)

        # check that we are recovering ~2000 files
        # assert len(fnames) > 2000, f"Looking for *.VH.nc filenames from the FTP server. There should be more than 2000 files found (Weekly VHI)"

        # check the first filename matches our expectations
        # assert fnames[0] == "VHP.G04.C07.NC.P1981035.VH.nc", f"The first filename should be Y: 1981 Wk: 035"

        # weeks = [int(_parse_time_from_filename(f)[-1]) for f in fnames]
        # years = [int(_parse_time_from_filename(f)[0]) for f in fnames]
        #
        # # check that our minimum years and max years are correct
        # assert (min(weeks) == 1) and (max(weeks) == 52), f"Week numbers should \
        # be between 1 and 52"
        # assert (min(years) == 1981) and (max(years) >= 2019), f"Year numbers \
        # should be between 1981 and greater than / equal to 2019"
        return


def test_dir_structure_created(tmp_path):
    VHIExporter(tmp_path)
    raw_folder = tmp_path / 'raw'
    raw_filename = 'VHP.G04.C07.NC.P1981035.VH.nc'

    expected_path = raw_folder / 'vhi/1981/VHP.G04.C07.NC.P1981035.VH.nc'
    filename = make_filename(raw_folder, raw_filename, 'vhi')

    assert expected_path == filename, f"Got {filename}, expected {expected_path}"
    assert raw_folder.exists(), f"`raw` directory should exist"
    assert (raw_folder / "vhi" / "1981").exists(), f"`raw/vhi/1981` directory should exist"
    return
