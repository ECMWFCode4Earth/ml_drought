""" https://curl.trillworks.com/

https://stackoverflow.com/questions/23102833/how-to-scrape-a-website-which-requires-login-using-python-and-beautifulsoup
"""

import os
import requests
import io
from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from tqdm import tqdm
import xarray as xr

import sys

sys.path.append("../..")
from scripts.utils import get_data_path
from scripts.drafts.cookies import cookies, headers


@dataclass
class GR4JDownload:
    """Download the GR4J Data for the study:

    Smith, K.A.; Tanguy, M.; Hannaford, J.; Prudhomme, C. (2018).
        Historic reconstructions of daily river flow for 303 UK catchments (1891-2015).
        NERC Environmental Information Data Centre.
        https://doi.org/10.5285/f710bed1-e564-47bf-b82c-4c2a2fe2810e

    @misc{smith2018,
        doi = {10.5285/f710bed1-e564-47bf-b82c-4c2a2fe2810e},
        url = {https://doi.org/10.5285/f710bed1-e564-47bf-b82c-4c2a2fe2810e},
        author = {Smith, K.A.;Tanguy, M.;Hannaford, J.;Prudhomme, C.},
        publisher = {NERC Environmental Information Data Centre},
        title = {Historic reconstructions of daily river flow for 303 UK catchments (1891-2015)},
        year = {2018}
    }

    Historic hydrological droughts 1891–2015: systematic characterisation for a diverse set of
     catchments across the UK.
    https://hess.copernicus.org/articles/23/4583/2019/

    This dataset is model output from the GR4J lumped catchment hydrology model.
     It provides 500 model realisations of daily river flow, in cubic metres per second (cumecs, m3/s),
     for 303 UK catchments for the period between 1891-2015. The modelled catchments are part of the
     National River Flow Archive (NRFA) (https://nrfa.ceh.ac.uk/) and provide good spatial coverage
     across the UK. These flow reconstructions were produced as part of the Research Councils UK (RCUK)
     funded Historic Droughts and IMPETUS projects, to provide consistent modelled daily flow data
     across the UK from 1891-2015, with estimates of uncertainty. This dataset is an outcome of the
     Historic Droughts Project (grant number: NE/L01016X/1).

    The GR4J model (v 1.0.2) was run over the calibration period (1982-2014) using 500,000 Latin
     Hypercube Sampled model parameter sets. These model parameters were assessed against observations
     from the National River Flow Archive (NRFA). For two catchments (the Thames at Kingston,
     and the Lea at Feildes Weir) the model was also calibrated against naturalised flows. The
     model was calibrated using a multi-objective approach comprising of 6 evaluation metrics: Nash
     Sutcliffe Efficiency (NSE), NSE on log flows (log NSE), Mean Absolute Percent Error (MAPE),
     Absolute Percent Bias (PBIAS), Absolute Percent Error in Mean Annual Minimum flows over a
     30 day accumulation period (MAM30), and Absolute Percent Error in the flow exceeded 95% of the time (Q95).

    The 500,000 model runs were then ranked by each evaluation metric, the ranks were summed, and the runs
     were reordered according to this final rank. Finally, in order to prevent uneven trade-offs between metrics,
     the runs were re-ordered according to thresholds of acceptability.

    Reconstructed flow timeseries were then run for the top 500 ranking model parameter sets, using PET
     (Potential Evapotranspiration) (Tanguy et al., 2017: doi https://doi.org/10.5285/17b9c4f7-1c30-4b6f-b2fe-f7780159939c),
     and reconstructed daily rainfall data, provided by the UK Met Office.

    The modelled data, and the supporting metadata files, were exported from the R software programme as
    comma separated value files (.csv), and ingested into the EIDC in this format.
    """

    data_dir: Path
    base_url: str = "https://catalogue.ceh.ac.uk/datastore/eidchub/f710bed1-e564-47bf-b82c-4c2a2fe2810e/"
    ensemble: bool = False
    download_simulation_csvs: bool = True

    # https://curl.trillworks.com/
    cookies: Optional[Dict[str, str]] = None
    headers: Optional[Dict[str, str]] = None

    def __post_init__(self):
        self.errors: List = []
        self._set_cookies_headers()
        if self.ensemble:
            #  (1) a 500-member ensemble of daily river flow time series for each catchment,
            #  with their corresponding model parameters and evaluation metric scores of
            #  model performance.
            self.url = self.base_url + "/Ensemble_500/"
        else:
            # (2) a single river flow time series (one corresponding to the top run of the 500),
            #  with the maximum and minimum daily limits of the 500 ensemble members.
            self.url = self.base_url + "/Single_Run/"

        #  make the output directory
        self.out_dir = (
            self.data_dir / "GR4J_data/ensemble"
            if self.ensemble
            else self.data_dir / "GR4J_data/single"
        )
        self.out_dir.mkdir(exist_ok=True, parents=True)

        #  Get list of all csvs
        self._get_list_of_csvs()

    def _set_cookies_headers(self):
        self.cookies = cookies
        self.headers = headers

    def _get_list_of_csvs(self):
        # find all of the csv files
        response = requests.get(self.url, headers=None, cookies=self.cookies)
        the_page = response.text
        soup = BeautifulSoup(the_page)

        # get the csv files from the path
        a_tags = [a_tag for a_tag in soup.find_all("a")]
        csvs = [a_tag.text for a_tag in a_tags if ".csv" in a_tag.text]
        self.metadata_csv = [csv for csv in csvs if "Metadata" in csv]
        self.simulation_csv = [csv for csv in csvs if not "Metadata" in csv]

    def _download_single_csv(self, csv_file: str) -> pd.DataFrame:
        csv_response = requests.get(
            self.url + csv_file, headers=None, cookies=self.cookies
        )
        assert (
            csv_response.status_code == 200
        ), f"Error downloading file: {self.url + csv_file}"
        return pd.read_csv(io.StringIO(csv_response.text))

    def download_csvs(
        self, list_of_csvs: List[str], fname: Optional[str] = None, desc: str = ""
    ):
        for csv_file in tqdm(list_of_csvs, desc=desc):
            out_file = (
                self.out_dir / fname if fname is not None else self.out_dir / csv_file
            )
            #  only download if not already downloaded
            if out_file.exists():
                continue

            # Download the file (if the cookies and headers work)
            try:
                df = self._download_single_csv(csv_file)
                df.to_csv(out_file)
            except AssertionError:
                self.errors.append(self.url + csv_file)

    def export(self):
        if self.download_simulation_csvs:
            self.download_csvs(list_of_csvs=self.simulation_csv, desc="Simulation CSVs")

        if self.ensemble:
            self.download_csvs(list_of_csvs=self.metadata_csv, desc="Metadata CSVs")
        else:
            self.download_csvs(
                list_of_csvs=self.metadata_csv,
                fname="ALL_results.csv",
                desc="Metadata CSVs",
            )

def read_gr4j_sims(data_dir: Path) -> Tuple[xr.Dataset, pd.DataFrame]:
    csvs = list((data_dir / "GR4J_data/single").glob("HD_FlowSingle*.csv"))
    metadata = pd.read_csv(list((data_dir / "GR4J_data/single").glob("ALL*.csv"))[0], index_col=0)

    # Read csvs into pandas dataframe
    gr4j_df = pd.concat([
        pd.read_csv(csv, index_col=0).rename(
            {"catchID": "station_id", "Date": "time", "Flow_Top_Calib": "sim",
                "Max_500": "ensemble_upper", "Min_500": "ensemble_lower"},
            axis=1
        ).astype({"time": "datetime64[ns]", "station_id": "int64"})
        for csv in csvs
    ])

    return gr4j_df.set_index(["station_id", "time"]).to_xarray(), metadata


if __name__ == "__main__":
    data_dir = get_data_path()
    data_dir = Path("/Volumes/Lees_Extend/data/")

    assert data_dir.exists()

    g = GR4JDownload(data_dir)
    g.export()
    print(f"Errors: {len(g.errors)}\n{g.errors}")
