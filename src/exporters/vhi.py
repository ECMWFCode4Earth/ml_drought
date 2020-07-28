from pathlib import Path
from typing import List, Tuple, Generator, Dict, Optional
import ftplib
from pprint import pprint
from functools import partial
import re
import pickle
import numpy as np
import warnings

from .base import BaseExporter

Pool = None  # this requires the pathos libary


class VHIExporter(BaseExporter):
    """Exports Vegetation Health Index from NASA site

    ftp.star.nesdis.noaa.gov
    """

    dataset = "vhi"

    def __init___(self, data_folder: Path = Path("data")) -> None:
        super().__init__(data_folder)

        global Pool
        if Pool is None:
            from pathos.multiprocessing import ProcessingPool as Pool

    @staticmethod
    def get_ftp_filenames(years: Optional[List]) -> List:
        """  get the filenames containing VHI """
        with ftplib.FTP("ftp.star.nesdis.noaa.gov") as ftp:
            ftp.login()
            ftp.cwd("/pub/corp/scsb/wguo/data/Blended_VH_4km/VH/")

            # append the filenames to a list
            listing: List = []
            ftp.retrlines("LIST", listing.append)
            # extract the filename
            filepaths = [f.split(" ")[-1] for f in listing]
            # extract only the filenames of interest
            vhi_files = [f for f in filepaths if ".VH.nc" in f]
            # extract only the years of interest
            if years is not None:
                years = [str(yr) for yr in years]
                vhi_files = [
                    f for f in vhi_files if any([f"P{yr}" in f for yr in years])
                ]
        return vhi_files

    @staticmethod
    def chunks(l: List, n: int) -> Generator:
        """ return a generator object which chunks list into sublists of size n
        https://chrisalbon.com/python/data_wrangling/break_list_into_chunks_of_equal_size/
        """
        # For item i in a range that is a length of l,
        for i in range(0, len(l), n):
            # Create an index range for l of n items:
            yield l[i : i + n]

    def save_errors(self, outputs: List) -> None:
        print("\nError: ", [errors for errors in outputs if errors is not None])

        # save the filenames that failed to a pickle object
        with open(self.raw_folder / "vhi_export_errors.pkl", "wb") as f:
            pickle.dump([error[-1] for error in outputs if error is not None], f)

    def _run_export(self, vhi_files: List, n_parallel_processes: int = 100) -> List:

        if n_parallel_processes > 1:
            pool = Pool(processes=n_parallel_processes)  # type: ignore

            # split the filenames into batches of 100
            batches = [batch for batch in self.chunks(vhi_files, 100)]

            # run in parallel for multiple file downloads
            args = dict(raw_folder=self.raw_folder)
            outputs = pool.map(partial(batch_ftp_request, args), batches)
        else:
            outputs = []
            batches = []
            for vhi_file in vhi_files:
                try:
                    batch_ftp_request(
                        args={"raw_folder": self.raw_folder}, filenames=[vhi_file]
                    )
                    batches.append([vhi_file])
                except Exception as e:
                    outputs.append(e)

        # write the output (TODO: turn into logging behaviour)
        print("\n\n*************************")
        print("VHI Data Downloaded")
        print("*************************")
        print("Errors:")
        pprint([error for error in outputs if error is not None])
        print(
            "Errors saved in data/raw/vhi_export_errors.pkl. Extract using \
            VHIExporter.check_failures()"
        )
        # save errors
        self.save_errors(outputs)

        return batches

    def check_failures(self) -> List:
        """ Read the outputted list of errors to the user """
        pickled_error_fname = "vhi_export_errors.pkl"
        assert (self.raw_folder / pickled_error_fname).exists(), (
            f'the file: {(self.raw_folder / "vhi_export_errors.pkl")} does not exist! '
            f"Required to check the files that failed"
        )

        with open(pickled_error_fname, "rb") as f:
            errors = pickle.load(f)

        return errors

    @staticmethod
    def get_default_years() -> List[int]:
        """ returns the default arguments for no. years """
        years = [yr for yr in range(1981, 2020)]
        return years

    @staticmethod
    def check_52_files(directory: Path, year: str) -> bool:
        files = [f for f in directory.glob("*.nc")]
        if (len(files) != 52) or (len(files) != 104):
            print(
                f"Not all files downloaded for {year}. Expected: [52 or 104] Got: {len(files)}"
            )
            return True
        else:
            return False

    def get_filepaths_for_year(self, year: str, vhi_files: List[str]) -> List[str]:
        filepaths = []
        # for every file on the FTP server
        for f in vhi_files:
            if _parse_time_from_filename(f)[0] == year:
                filepaths.append(f)
        return filepaths

    def get_missing_filepaths(self, vhi_files: List[str]) -> List[str]:
        # get the missing filepaths if the number of files != 52
        year_paths = [f for f in (self.raw_folder / "vhi").glob("*")]
        years = [y.as_posix().split("/")[-1] for y in year_paths]

        missing_filepaths = []
        for year, dir in zip(years, year_paths):
            if self.check_52_files(dir, year):
                missing_filepaths.extend(self.get_filepaths_for_year(year, vhi_files))

        return missing_filepaths

    def export(
        self,
        years: Optional[List] = None,
        repeats: int = 5,
        n_parallel_processes: int = 1,
        check_exists: bool = True,
    ) -> List:
        """Export VHI data from the ftp server.
        By default write output to raw/vhi/{YEAR}/{filename}

        Arguments:
        ---------
        years : Optional[List] = None
            list of years that you want to download. If None, all years will
            be downloaded
        repeats: int = 5
            The number of times to retry downloads which failed
        n_parallel_processes: int = 100
            The number of processes to run. If 1, the download happens serially
        check_exists: bool = True
            Check whether the file has already been downloaded, if so skip it.

        Returns:
        -------
        batches : List
            list of lists containing batches of filenames downloaded
        """
        if years is not None:
            assert min(years) >= 1981, (
                f"Minimum year cannot be less than 1981. " f"Currently: {min(years)}"
            )
            if max(years) > 2020:
                warnings.warn(
                    f"Non-breaking change: max(years) is:{ max(years)}. "
                    f"But no files later than 2019"
                )

        # get the filenames to be downloaded
        vhi_files = self.get_ftp_filenames(years)

        # TODO: check that vhi files don't already exist
        if check_exists:
            vhi_files = np.array(sorted(vhi_files))
            years_of_files: List[str] = [
                _parse_time_from_filename(vhi_file)[0] for vhi_file in vhi_files
            ]
            exists_bool = [
                (self.output_folder / years_of_files[ix] / filename).exists()
                for ix, filename in enumerate(vhi_files)
            ]
            if sum(exists_bool) > 0:
                print(  # type: ignore
                    f"{sum(exists_bool)} VHI Files already downloaded. Skipping:",
                    vhi_files[exists_bool],  # type: ignore
                )  # type: ignore

            # convert back to a list and drop the already existing files
            vhi_files = list(vhi_files[~np.array(exists_bool)])

        # run the download steps in parallel
        batches = self._run_export(vhi_files, n_parallel_processes)

        for _ in range(repeats):
            missing_filepaths = self.get_missing_filepaths(vhi_files)
            batches = self._run_export(missing_filepaths, n_parallel_processes)
            print(f"**{_} of {repeats} VHI Downloads completed **")

        return batches


# ------------------------------------------------------------------------------
# Parallel functions
# ------------------------------------------------------------------------------


def _parse_time_from_filename(filename) -> Tuple:
    # regex pattern (4 digits after '.P')
    year_pattern = re.compile(r".P\d{4}")
    # extract the week_number
    week_num = year_pattern.split(filename)[-1].split(".")[0]
    # extract the year from the filename
    year = year_pattern.findall(filename)[0].split(".P")[-1]

    return year, week_num


def make_filename(raw_folder: Path, raw_filename: str, dataset: str = "vhi") -> Path:
    # check that the string is a legitimate name
    assert (
        len(raw_filename.split("/")) == 1
    ), f" \
        filename cannot have subdirectories in it '/'. Must be the root \
        filename. Currently: {raw_filename}\
        "

    # make the dataset folder ('VHI')
    dataset_folder = raw_folder / dataset
    if not dataset_folder.exists():
        dataset_folder.mkdir()

    # make the year folder
    year = _parse_time_from_filename(raw_filename)[0]
    assert isinstance(
        year, str
    ), f"year must be a string! currently: < {year}, {type(year)} >"
    year_folder = dataset_folder / year
    if not year_folder.exists():
        year_folder.mkdir()

    # make the filename e.g. 'raw/vhi/1981/VHP.G04.C07.NC.P1981035.VH.nc'
    filename = year_folder / raw_filename

    return filename


def download_file_from_ftp(
    ftp_instance: ftplib.FTP, filename: str, output_filename: Path
) -> None:
    # check if already exists
    if output_filename.exists():
        print(f"File already exists! {output_filename}")
        return

    # download to output_filename
    with output_filename.open("wb") as out_f:
        ftp_instance.retrbinary("RETR " + filename, out_f.write)

    if output_filename.exists():
        print(f"Successful Download! {output_filename}")
    else:
        print(f"Error Downloading file: {output_filename}")


def batch_ftp_request(args: Dict, filenames: List) -> None:
    # unpack multiple arguments
    raw_folder = args["raw_folder"]

    # create one FTP connection for each batch
    with ftplib.FTP("ftp.star.nesdis.noaa.gov") as ftp:
        ftp.login()
        ftp.cwd("/pub/corp/scsb/wguo/data/Blended_VH_4km/VH/")

        # download each filename using this FTP object
        for raw_filename in filenames:
            output_filename = make_filename(raw_folder, raw_filename, dataset="vhi")
            download_file_from_ftp(ftp, raw_filename, output_filename)
