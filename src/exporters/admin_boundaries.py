from pathlib import Path
import os
from typing import Dict

from .base import BaseExporter


class OCHAExporter(BaseExporter):
    """Export Administrative boundaries for country from the
    Organisation for the Coordination of Humanitarian Affairs.

    https://data.humdata.org/
    """

    country: str

    def __init__(self, data_folder: Path = Path("data")) -> None:
        super().__init__(data_folder)
        # write the download to landcover
        self.admin_bounds_folder = self.raw_folder / "boundaries" / self.country
        if not self.admin_bounds_folder.exists():
            self.admin_bounds_folder.mkdir(parents=True, exist_ok=True)

    def wget_file(self, url_path: str) -> None:
        output_file = self.admin_bounds_folder / url_path.split("/")[-1]
        if output_file.exists():
            print(f"{output_file} already exists! Skipping")
            return None

        os.system(f"wget {url_path} -P {self.admin_bounds_folder.as_posix()}")

    def unzip(self, fname: Path) -> None:
        print(f"Unzipping {fname.name}")

        os.system(
            f"unzip {fname.as_posix()} -d {self.admin_bounds_folder.resolve().as_posix()}"
        )
        print(f"{fname.name} unzipped!")

    def rename_dirs(self, fname, new_fname) -> None:
        """TODO: need to rename the unzipped file NOT the zipped file"""
        new_fname = new_fname + ".zip" if new_fname[-4:] != ".zip" else new_fname
        new_filepath = fname.absolute().parents[0] / new_fname
        os.system(f"mv {fname.absolute().as_posix()} {new_filepath.as_posix()}")
        print(f"Renamed file to: {new_fname}")


class KenyaAdminExporter(OCHAExporter):
    country = "kenya"

    urls: Dict = {
        "kenya-admin-level-1-boundaries": "https://data.humdata.org"
        "/dataset/8158e5a1-065b-47b7-b5ad-aff3a7157535/resource/"
        "9dbe8e4f-3085-467b-8111-402ba9ceaadc/download/ken_admin1.zip",
        "kenya-admin-level-2-boundaries": "https://data.humdata.org"
        "/dataset/23b3ea32-5cdc-47a0-94a5-763a611a2449/resource/"
        "65b395f2-4ce6-41d9-a169-d17e021a7843/download/kendistricts.zip",
        "kenya-admin-level-3-wards": "https://data.humdata.org"
        "/dataset/db0ba5f4-cd9a-4512-b9ca-3c79f59fed08/resource/"
        "2fe6fcaa-58af-4d5d-a17d-81c73574ae2a/download/kenya_wards.zip",
        "kenya-admin-level-3-boundaries": "https://data.humdata.org"
        "/dataset/1c6caac8-874e-47fd-87c4-4d5d91c47e8f/resource/"
        "321ce757-f063-475a-9442-c181c444e429/download/kendivisions.zip",
        "kenya-admin-level-4-boundaries": "https://data.humdata.org"
        "/dataset/28d9cc75-2b66-4901-b3e6-cff64907a083/resource/"
        "c29244ca-2740-4700-829f-7b54aedb6f8d/download/kenlocations.zip",
        "kenya-admin-level-5-boundaries": "https://data.humdata.org"
        "/dataset/ef819e1b-eda0-4db9-af36-adac33016973/resource/"
        "40b33c69-a47a-47f2-8503-4ddf142127c9/download/kensublocations.zip",
    }

    def export(self) -> None:
        """Export functionality for the Kenya Admin Boundaries as .shp files
        """

        for new_fname, url in zip(self.urls.keys(), self.urls.values()):
            fname = url.split("/")[-1]
            # check if the file already exists
            if (self.admin_bounds_folder / fname).exists():
                print("Data already downloaded!")

            else:
                self.wget_file(url_path=url)
                self.unzip(fname=(self.admin_bounds_folder / fname))


class EthiopiaAdminExporter(OCHAExporter):
    country = "ethiopia"
    urls: Dict = {
        "ethiopia-admin-boundaries": "https://data.humdata.org"
        "/dataset/a5788674-b4d1-4731-9ec6-45040a7be731/resource/"
        "91274071-1a6d-4cce-a7d3-c29274a4b249/download/ethiopia.zip"
    }

    def export(self) -> None:
        """Export functionality for the Ethiopia Admin Boundaries as .shp files
        """
        for new_fname, url in zip(self.urls.keys(), self.urls.values()):
            fname = url.split("/")[-1]
            # check if the file already exists
            if (self.admin_bounds_folder / fname).exists():
                print("Data already downloaded!")

            else:
                self.wget_file(url_path=url)
                self.unzip(fname=(self.admin_bounds_folder / fname))
                self.rename_dirs(
                    fname=(self.admin_bounds_folder / fname), new_fname=new_fname
                )


class EastAfricaAdminExporter(OCHAExporter):
    country = "ethiopia"

    urls: Dict = {
        "east-africa-boundaries": "https://data.humdata.org"
        "/dataset/756f07a8-e462-49a9-ab89-ff481d586953/resource/"
        "50435c12-b850-4b07-8c1e-b2368dea085b/download/esa_admin1_region.rar"
    }

    def export(self) -> None:
        """Export functionality for the Ethiopia Admin Boundaries as .shp files
        """
        assert False, (
            "Currently no way of extracting .rar files without"
            "having sudo control - `sudo apt-get install unrar`"
            "https://www.tecmint.com/how-to-open-extract-and-create-rar-files-in-linux/"
        )

        for url in self.urls.values():
            fname = url.split("/")[-1]
            # check if the file already exists
            if (self.admin_bounds_folder / fname).exists():
                print("Data already downloaded!")

            else:
                self.wget_file(url_path=url)
                self.unzip(fname=(self.admin_bounds_folder / fname))
