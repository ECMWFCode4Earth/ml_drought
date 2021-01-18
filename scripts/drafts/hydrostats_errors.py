from typing import Dict, List, Optional
from tqdm import tqdm
import hydrostats as hs
import pandas as pd
import xarray as xr
import numpy as np

if __name__ == "__main__":
    all_result_dfs: List[pd.DataFrame] = []

    for ix, station_id in tqdm(
        enumerate(lstm_preds.station_id.values, desc="Calculating Metrics")
    ):
        # station_id = lstm_preds.station_id.values[0]
        df = (
            lstm_preds.sel(station_id=station_id)
            .to_dataframe()
            .drop("station_id", axis=1)
        )

        epsilon = 1e-10

        result_df = hs.make_table(
            merged_dataframe=df,
            metrics=["NSE", "KGE (2012)", "MAPE"],
            seasonal_periods=[
                ["12-01", "02-29"],
                ["03-01", "05-31"],
                ["06-01", "08-31"],
                ["09-01", "11-30"],
            ],
            remove_neg=True,
            remove_zero=False,
            location=station_id,
        )

        inv_kge_df = hs.make_table(
            merged_dataframe=(1 / df + epsilon),
            metrics=["KGE (2012)"],
            seasonal_periods=[
                ["12-01", "02-29"],
                ["03-01", "05-31"],
                ["06-01", "08-31"],
                ["09-01", "11-30"],
            ],
            remove_neg=True,
            remove_zero=False,
            location=station_id,
        ).rename({"KGE (2012)": "invKGE"}, axis=1)
        log_nse_df = hs.make_table(
            merged_dataframe=np.log(df + epsilon),
            metrics=["NSE"],
            seasonal_periods=[
                ["12-01", "02-29"],
                ["03-01", "05-31"],
                ["06-01", "08-31"],
                ["09-01", "11-30"],
            ],
            remove_neg=True,
            remove_zero=False,
            location=station_id,
        ).rename({"NSE": "logNSE"}, axis=1)

        #  join all error metrics together
        result_df = pd.concat(
            [
                result_df,
                inv_kge_df.drop("Location", axis=1),
                log_nse_df.drop("Location", axis=1),
            ],
            axis=1,
        )

        #  rename columns/rows
        result_df = result_df.rename(
            {
                "Full Time Series": "All",
                "December-01:February-29": "DJF",
                "March-01:May-31": "MAM",
                "June-01:August-31": "JJA",
                "September-01:November-30": "SON",
            }
        ).rename({"Location": "station_id", "KGE (2012)": "KGE"}, axis=1)
        result_df = result_df.reset_index().rename({"index": "period"}, axis=1)
        all_result_dfs.append(result_df)

        if ix == 5:
            break
