from typing import Tuple 
from typing import Optional, Dict
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import xarray as xr
import seaborn as sns 
import matplotlib.pyplot as plt


def get_title_nse_scores(nse_df: pd.DataFrame, station_id) -> str:
    col = nse_df.columns[0]
    title_str = f"{col}: {nse_df.loc[station_id, col]:.2f}"

    for col in nse_df.columns[1:]:
        title_str += f" -- {col}: {nse_df.loc[station_id, col]:.2f}"

    return title_str 


def get_hydrological_year(year: int) -> Tuple[pd.Timestamp]:
    """return Oct-Oct https://nrfa.ceh.ac.uk/news-and-media/news/happy-new-water-year"""
    min_ts = pd.Timestamp(day=10, month=10, year=year - 1)
    max_ts = pd.Timestamp(day=30, month=9, year=year)
    return (min_ts, max_ts)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def get_all_data_for_basins(all_preds: xr.Dataset, station_int: int) -> pd.DataFrame:
    columns = ["obs", "LSTM", "EALSTM", "TOPMODEL", "ARNOVIC", "PRMS", "SACRAMENTO"]
    df = all_preds.sel(station_id=station_int)[columns].to_dataframe()
    data = df.reset_index().set_index("station_id")

    def fixC(colname: str) -> str:
        cname = colname.replace("SimQ_", "")
    #     cname = "VIC" if cname == "ARNOVIC" else cname
    #     cname = "Sacramento" if cname == "SACRAMENTO" else cname
        return cname

    data.columns = [fixC(c) for c in data.columns]


def plot_station_hydrograph(
    data: pd.DataFrame, static_df: pd.DataFrame, dynamic: xr.Dataset, station_id: int, nse_df: pd.DataFrame, plot_conceptual: bool = True, ax: Optional = None,
    legend_kwargs: Dict = {}, legend: bool = True, nse_in_title: bool = False,
    non_overlap: bool = False,
):
    assert all(np.isin(["time", "LSTM", "EALSTM", "obs"], data.columns))
    station_name = static_df.loc[station_id, "gauge_name"]
    precip = dynamic.sel(time=data["time"].values, station_id=station_id)["precipitation"].drop("station_id", axis=1)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    else:
        fig = plt.gcf()
        
    ax.plot(data["time"], data["LSTM"], color=sns.color_palette()[0], label="LSTM", alpha=1, linewidth=2)
    ax.plot(data["time"], data["EALSTM"], color=sns.color_palette()[1], label="EA LSTM", alpha=1, linewidth=2)
    if plot_conceptual:
        ax.plot(data["time"], data["TOPMODEL"], label="TOPMODEL", alpha=0.5, linewidth=1, color=sns.color_palette()[2])
        ax.plot(data["time"], data["ARNOVIC"], label="VIC", alpha=0.5, linewidth=1, color=sns.color_palette()[3])
        ax.plot(data["time"], data["PRMS"], label="PRMS", alpha=0.5, linewidth=1, color=sns.color_palette()[4])
        ax.plot(data["time"], data["SACRAMENTO"], label="Sacramento", alpha=0.5, linewidth=1, color=sns.color_palette()[5])

    ax.plot(data["time"], data["obs"], color="k", ls=":", label="Observed")
    if legend:
        ax.legend(**legend_kwargs)
        
    if non_overlap:
        xmin = ax.get_ylim()[0]
        xmax = ax.get_ylim()[1] + (1.5 * data["LSTM"].std())
        ax.set_ylim(xmin, xmax)
    
    # Plot the rainfall too
    ax2 = ax.twinx()
    ax2.bar(data["time"], precip, alpha=0.4)
    ax2.set_ylim([0, precip.max() + 5*precip.std()])
    # ax2.set_yticklabels([])
    # ax2.set_yticks([])
    ax2.set_ylabel("Precipitation [mm day$^{-1}$]")
    ax2.invert_yaxis()
    if non_overlap:
        xmin2 = ax2.get_ylim()[0]
        xmax2 = ax2.get_ylim()[1] + (1.5 * np.std(precip))
        ax2.set_ylim(xmin2, xmax2)
    
    title = f"Station: {station_name} - {station_id}"
    if nse_in_title:
        title = title + "\n" + get_title_nse_scores(nse_df, station_id)

    ax.set_title(title)  
    ax.set_xlabel("Time")
    ax.set_ylabel("Specific Discharge [mm day$^{-1}$ km$^{2}$]")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    # rotate and align the tick labels so they look better
    fig.autofmt_xdate()

    # sns.despine()
    
    return f, ax


if __name__ == "__main__":
    year = "2007"
    hydro_year_ts = get_hydrological_year(int(year))
    station_int = 47001
    d = get_all_data_for_basins(all_preds, station_int).set_index("time")

    f, ax = plt.subplots(figsize=(12, 4))
    plot_station_hydrograph(
        data=d.loc[hydro_year_ts[0]: hydro_year_ts[-1]].reset_index(), 
        station_id=station_int, 
        static_df=static_df,
        dynamic=dynamic,
        nse_df=all_metrics["nse"], 
        legend_kwargs={"loc": "lower left"}, 
        ax=ax, 
        legend=True, 
        nse_in_title=True, 
        non_overlap=True
    )
