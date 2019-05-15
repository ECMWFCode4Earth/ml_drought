"""
functionality:
1) download VHI data in parallel
2)

CDS functions:
- create_area()
- _filename_from_selection_request()
- make_filename()
- _print_api_request()
- _export()
- get_era5_times()
- get_dataset()
- _correct_input()
- _check_iterable()
- create_selection_request()
- export()
"""

from pathlib import Path
from typing import Dict, Optional, List
from ftplib import FTP
import multiprocessing
from pprint import pprint
import re

# unnecessary extra dependency?
from pathos.multiprocessing import ProcessingPool as Pool

from .base import BaseExporter, Region

# TODO: separate into general ftp exporter class?
# class FTPExporter():
#     """"""

class VHIExporter(BaseExporter):
    """Exports Vegetation Health Index from NASA site

    ftp.star.nesdis.noaa.gov
    """

    def __init__(self, data_folder: Path = Path('data')) -> None:
        super().__init__(data_folder)

    def get_ftp_filenames() -> List:
        """  get the filenames containing VHI """
        with FTP('ftp.star.nesdis.noaa.gov') as ftp:
            ftp.login()
            ftp.cwd('/pub/corp/scsb/wguo/data/Blended_VH_4km/VH/')

            # append the filenames to a list
            listing = []
            ftp.retrlines("LIST", listing.append)
            # extract the filename
            filepaths = [f.split(' ')[-1] for f in listing]
            # extract only the filenames of interest
            vhi_files = [f for f in filepaths if ".VH.nc" in f]

        return vhi_files

    @staticmethod
    def chunks(l, n):
        # return a generator object which chunks list into sublists of size n
        # https://chrisalbon.com/python/data_wrangling/break_list_into_chunks_of_equal_size/
        # For item i in a range that is a length of l,
        for i in range(0, len(l), n):
            # Create an index range for l of n items:
            yield l[i:i+n]

    def _parse_time_from_filename(filename):
        # regex pattern (4 digits after '.P')
        year_pattern = re.compile('.P\d{4}')
        # extract the week_number
        week_num = year_pattern.split(filename)[-1].split('.')[0]
        # extract the year from the filename
        year = year_pattern.findall(filename)[0].split('.P')[-1]

        return year, week_num

    def make_filename(self, raw_filename: str, dataset: str = 'vhi',) -> Path:
        # check that the string is a legitimate name
        assert len(filename.split('/')) == 1, f"filename cannot have subdirectories in it '/'. Must be the root filename. Currently: {filename}"

        # make the dataset folder ('VHI')
        dataset_folder = self.raw_folder / dataset
        if not dataset_folder.exists():
            dataset_folder.mkdir()

        # make the year folder
        year = _parse_time_from_filename(filename)[0]
        assert isinstance(year, str), f"year must be a string! currently: < {year}, {type(year)} >"
        year_folder = dataset_folder / year
        if not year_folder.exists():
            year_folder.mkdir()

        # make the filename e.g. 'raw/vhi/1981/VHP.G04.C07.NC.P1981035.VH.nc'
        filename = year_folder / raw_filename

        return filename

    def download_file_from_ftp(ftp_instance: FTP,
                               filename: str,
                               output_filename: Path) -> None:
        # check if already exists
        if output_filename.exists():
            print(f"File already exists! {output_filename}")
            return

        # download to output_filename
        with open(output_filename,'wb') as out_f:
            ftp_instance.retrbinary("RETR " + filename, out_f.write)

        if output_filename.exists():
            print(f"Successful Download! {output_filename}")
        else:
            print(f"Error Downloading file: {output_filename}")

        return

    def batch_ftp_request(filenames: List) -> None:
        # create one FTP connection for each batch
        with FTP('ftp.star.nesdis.noaa.gov') as ftp:
            ftp.login()
            ftp.cwd('/pub/corp/scsb/wguo/data/Blended_VH_4km/VH/')

            # download each filename using this FTP object
            for filename in filenames:
                output_filename = (
                    self.make_filename(filename, dataset='vhi')
                )
                download_file_from_ftp(ftp, filename, output_filename)

        return

    def save_errors(outputs: List) -> None:
        print("\nError: ",[errors for errors in outputs if errors != None])

        # save the filenames that failed to a pickle object
        with open(self.raw_folder / 'vhi_export_errors.pkl','wb') as f:
            pickle.dump([error[-1] for error in outputs if error != None], f)

        return

    def run_parallel(vhi_files: List) -> List:
        pool = Pool(processes=100)

        # split the filenames into batches of 100
        batches = [batch for batch in chunks(vhi_files,100)]

        # run in parallel for multiple file downloads
        outputs = pool.map(batch_ftp_request, batches)

        # write the output (TODO: turn into logging behaviour)
        print("\n\n*************************")
        print("VHI Data Downloaded")
        print("*************************")
        print("Errors:")
        pprint([ri for ri in ris if ri != None])
        # save errors
        save_errors(outputs)

        return batches


    def export() -> Path:
        """Export VHI data from the ftp server
        """
        vhi_files = get_ftp_filenames()

        batches = run_parallel(vhi_files)

        return
