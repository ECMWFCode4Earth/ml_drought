import numpy as np
import pandas as pd
import xarray as xr
from typing import Tuple, Optional, Dict, Union, List, Any
from pathlib import Path
import calendar
import seaborn as sns
import torch
import matplotlib.pyplot as plt
import sys
import tqdm

sys.path.append("../..")

from scripts.utils import get_data_path
from src.models import load_model

# ---------------------------------------------------------
# Extract the static embeddings from the EALSTM
# ---------------------------------------------------------
from src.models.data import DataLoader


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def build_static_x(
    x: Tuple[np.array], ealstm, expected_size: Optional[Tuple[float, float]] = None
) -> Tuple[List[np.array], List[np.array], List[np.array]]:
    """From the x TrainData object (stored as a Tuple) and the ealstm model,
    calculate the static_data that is passed through the static embedding.
    """
    all_static_x = []
    all_latlons = []
    all_pred_months = []

    for i in range(len(x)):  #  EACH BATCH (of X, y pairs)
        # the components of the static data
        pred_month_data = x[i][1]
        latlons_data = x[i][2]
        yearly_aggs_data = x[i][4]
        static_data = x[i][5]

        # expected size of the static data ?
        if i == 0:
            print(f"Current size of static data: {static_data.shape[-1]}")
        if expected_size is not None:
            assert static_data.shape[-1] == expected_size, f"{static_data.shape}"

        # append the static_arrays
        static_x = []
        # normalise latlon
        static_x.append(
            (latlons_data - latlons_data.mean(axis=0)) / latlons_data.std(axis=0)
        )  # 0, 1
        static_x.append(yearly_aggs_data)  # 2: 9
        static_x.append(static_data)

        # one_hot_encode the pred_month_data
        # ONLY if pred_month is included in ealstm
        if ealstm.pred_month_static:
            try:
                static_x.append(
                    ealstm._one_hot(torch.from_numpy(pred_month_data), 12).numpy()
                )
            except TypeError:
                static_x.append(
                    ealstm._one_hot(torch.from_numpy(pred_month_data), 12).cpu().numpy()
                )

        # exclude Nones
        static_x = np.concatenate([x for x in static_x if x is not None], axis=-1)
        #  print("Static X Data Shape: ", static_x.shape)

        # all data
        all_static_x.append(static_x)

        # metadata (latlons and pred_months)
        all_latlons.append(latlons_data)
        all_pred_months.append(pred_month_data)

    return all_static_x, all_latlons, all_pred_months


def calculate_embeddings(static_x: np.ndarray, W: np.ndarray, b: np.array) -> np.array:
    """Calculate the static embedding from the input data
    and the input gate weights and biases.

    embedding = sigmoid ( WX + b )
    """
    assert (
        W.T.shape[0] == static_x.shape[-1]
    ), f"Matrix operations must be valid {static_x.shape} * {W.T.shape}"

    embedding = []
    for pixel_ix in range(static_x.shape[0]):
        embedding.append(sigmoid(np.dot(W, static_x[pixel_ix]) + b))
    return np.array(embedding)


def get_static_embedding(
    ealstm,
) -> Tuple[List[np.array], Tuple[List[np.array], np.array, List[np.array]]]:
    """Use the EALSTM dataloader to read in the training data and create
    the static embedding.
    """
    # get W, b from state_dict
    od = ealstm.model.static_embedding.state_dict()
    try:
        W = od["weight"].numpy()
        b = od["bias"].numpy()
    except TypeError:
        W = od["weight"].cpu().numpy()
        b = od["bias"].cpu().numpy()

    # get X_static data from dataloader
    print("Calling Training DataLoader")
    dl = ealstm.get_dataloader("train", batch_file_size=1, shuffle_data=False)
    x = [x for (x, y) in dl]

    # build static_x matrix
    all_static_x, all_latlons, all_pred_months = build_static_x(  # type: ignore
        x, ealstm
    )
    # check w^Tx + b is a valid matrix operation
    assert (
        W.T.shape[0] == all_static_x[0].shape[-1]
    ), f"W.T shape: {W.T.shape} static_x shape: {all_static_x[0].shape}"

    # calculate the embeddings
    all_embeddings = []
    for static_x in all_static_x:
        embedding = calculate_embeddings(static_x, W=W, b=b)
        all_embeddings.append(embedding)

    return (
        all_embeddings,
        (all_static_x, np.array(all_latlons), np.array(all_pred_months)),
    )


# ---------------------------------------------------------
# Cluster the Static Embeddings
# ---------------------------------------------------------

from collections import defaultdict
from typing import List, Dict, Union
from sklearn.cluster import KMeans


def fit_kmeans(
    array: np.array, ks: List[int] = [4], init_array: Optional[np.ndarray] = None
) -> Tuple[Dict[int, Dict[int, int]], Dict[int, KMeans]]:
    """Fit k-means with multiple values for `k` and return
    the clusters (1:k) predicted for each pixel as a dictionary.
    """
    # initialise the output dictionary
    clusters: Dict[int, Dict[int, int]] = {k: {} for k in ks}
    estimators: Dict[int, KMeans] = {}

    for k in ks:
        if init_array is not None:
            assert init_array.shape[0] == k, "First dimension should be"
            clusterer = KMeans(
                n_clusters=k, random_state=0, init=init_array, n_init=1
            ).fit(array)
        else:
            clusterer = KMeans(
                n_clusters=k, random_state=0, init="k-means++", n_init=200
            ).fit(array)

        for pixel in range(array.shape[0]):
            arr = array[pixel, :]
            clusters[k][pixel] = clusterer.predict(arr.reshape(1, -1))[0]

        estimators[k] = clusterer

    return clusters, estimators


def convert_clusters_to_ds(
    ks: List[int],
    static_clusters: Dict[int, np.array],
    pixels: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    time: Union[pd.Timestamp, int] = 1,
) -> xr.Dataset:
    """Create an xr.Dataset object from the output of the static
    embedding clustering. Allows for easy plotting, subsetting
    and all the other goodness of xarray objects.
    """
    out = []
    for k in ks:
        cluster = np.array([v for v in static_clusters[k].values()])
        coords = {"pixel": pixels}
        dims = ["pixel"]
        cluster_ds = xr.Dataset(
            {
                f"cluster_{k}": (dims, cluster),
                "lat": (dims, latitudes),
                "lon": (dims, longitudes),
                "time": (dims, [time for _ in range(len(latitudes))]),
            }
        )
        out.append(cluster_ds)

    static_cluster_ds = xr.auto_combine(out)
    static_cluster_ds = (
        static_cluster_ds.to_dataframe().set_index(["time", "lat", "lon"]).to_xarray()
    )

    return static_cluster_ds


def plot_cluster_ds(
    ks: List[int], static_cluster_ds: xr.Dataset, cmap=None, month_abbr: str = ""
):
    """For each `k` plot the predicted clusters.
    """
    for k in ks:
        fig, ax = plt.subplots(figsize=(12, 8))
        static_cluster_ds[f"cluster_{k}"].plot(ax=ax, cmap=cmap)
        ax.set_title(f"Output of Static Embedding Clustering [k={k}]\n{month_abbr}")

        for item in [ax.title, ax.xaxis.label, ax.yaxis.label]:
            item.set_fontsize(20)


# ---------------------------------------------------------
# Plotting the Static Embeddings
# ---------------------------------------------------------
from matplotlib.colors import ListedColormap
from src.utils import Region
from src.preprocess.utils import select_bounding_box

# ----------------------------
# Functions
# ----------------------------


def get_matching_groups(
    reference_da: xr.DataArray,
    comparison_da: xr.DataArray,
    percent: bool = True,
    regions: Optional[List[Region]] = None,
) -> Tuple[Dict[float, float], pd.DataFrame]:
    # get the unique values from the reference_da
    group_vals = np.unique(reference_da.values[~np.isnan(reference_da.values)])

    if regions is not None:
        df = count_mappings_for_regions(reference_da, comparison_da, regions)
        remap_dict = match_by_region_algorithm(df, regions)

    else:
        # calculate the number of matching pixels
        df = convert_counts_dict_to_dataframe(
            count_matching_pixels(reference_da, comparison_da)
        )

        # calculate_the remap_dict
        remap_dict = most_overlapping_pixels_algorithm(df, percent=percent)

    # check that the groups are matching / all groups are included
    assert all(np.isin(np.unique(df["reference_group"].values), group_vals))
    assert all(np.isin(np.unique(df["comparison_group"].values), group_vals))

    # check all values in group_vals are in the dict keys
    assert all(np.isin([k for k in remap_dict.keys()], group_vals))

    return remap_dict, df


def count_matching_pixels(
    reference_da: xr.Dataset, comparison_da: xr.Dataset
) -> Dict[float, Dict[float, float]]:
    """Count the number of pixels for each value
        in comparison_da for each reference value
        in reference_da

    Returns:
    -------
     Dict[float, Dict[float, float]]
        keys = reference_da values "group_0"
        values = {comparison_da values: count of matches} "group_1"
    """
    unique_counts = dict()

    # for each reference value in reference_da
    # excluding np.nan
    for value in np.unique(reference_da.values[~np.isnan(reference_da.values)]):
        # get the pixels from Comparison corresponding to `value` in Reference
        np_arr = comparison_da.where(reference_da == value).values
        # drop nans from matching values
        np_arr = np_arr[~np.isnan(np_arr)]
        # calculate the number of group_1 pixels
        counts = np.unique(np_arr, return_counts=True)
        unique_counts[value] = dict(zip(counts[0], counts[1]))

    return unique_counts


def convert_counts_dict_to_dataframe(unique_counts: dict) -> pd.DataFrame:
    """create long format dataframe from counts in unique_counts
    reference_da = group_0
    comparison_da = group_1
    """
    df = pd.DataFrame(unique_counts)  # rows = group_1_values, cols = group_0_values
    df.columns = df.columns.rename("reference_group")
    df.index = df.index.rename("comparison_group")
    # 2D -> 1D idx, group_0, group_1, count
    df = df.unstack().reset_index().rename(columns={0: "count"})

    counts = df.groupby("comparison_group")["count"].sum()
    df["pct"] = df.apply(
        lambda x: x["count"] / counts.loc[x["comparison_group"]], axis=1
    )

    return df


def get_max_count_row(df: pd.DataFrame) -> pd.Series:
    """Get the row with the largest count from df"""
    return df.loc[df["count"].idxmax()]


def get_max_percent_row(df: pd.DataFrame) -> pd.Series:
    """Get the row with the highest percentage overlap"""
    return df.loc[df["pct"].idxmax()]


def coarsen_da(ds: xr.Dataset, rolling_size: int = 3) -> xr.Dataset:
    # return ds.coarsen(lat=5, lon=5).median()
    return ds.rolling(lat=rolling_size).median().rolling(lon=rolling_size).median()


def drop_already_assigned_values(
    df: pd.DataFrame, assigned_group_values: List[float]
) -> pd.DataFrame:
    """drop the values that have been assigned (added to the lists)
    """
    # remove the matches from group_1 / comparison_group
    df = df.loc[~np.isin(df["comparison_group"], assigned_group_values)]
    # remove the matches from group_0 / reference_group
    df = df.loc[~np.isin(df["reference_group"], assigned_group_values)]
    return df


def calculate_remap_dict(
    reference_group_list: List[float], comparison_group_list: List[float]
) -> Dict[float, float]:
    """create dictionary object containing the mapping from reference_group -> comparison_group"""
    remap_dict = dict()
    #  TODO: Tis assumption is not true it's not symmetrical
    # remap dict is symmetrical:
    # values in group_0->group_1 are the same mapping as group_1 -> group_0
    remap_dict.update(dict(zip(reference_group_list, comparison_group_list)))
    remap_dict.update(dict(zip(remap_dict.values(), remap_dict.keys())))

    # sort the remap_dict
    remap_dict = {k: remap_dict[k] for k in sorted(remap_dict)}

    return remap_dict


def most_overlapping_pixels_algorithm(
    df: pd.DataFrame, percent: bool = True
) -> Dict[float, float]:
    """match the 'closest' group from reference_group_list in comparison_group_list"""

    assert all(
        np.isin(["reference_group", "comparison_group"], [c for c in df.columns])
    ), f"Need columns reference_group comparison_group. Found: {df.columns}"
    # get the counts of each pixel value/group (excl. nans) and select the most cross-overs (by percentages)
    # order is important so we do the BEST match first
    # get the largest first
    reference_group_list = []
    comparison_group_list = []

    # match each reference_group to closest matching comparison_group
    # track progress by removing the matches that have already been made
    # from the dataframe
    while df.shape[0] > 0:
        # IF only one group-value left, assign it to the final remaining group (itself)
        if len(df["comparison_group"].unique()) == 1:
            # final value is itself
            #  remap_dict[df['comparison_group'].unique()[0]] = df['comparison_group'].unique()[0]
            reference_group_list.append(df["comparison_group"].unique()[0])
            comparison_group_list.append(df["comparison_group"].unique()[0])
            df = drop_already_assigned_values(
                df, reference_group_list + comparison_group_list
            )

        else:
            # otherwise match to the closest remaining match (most overlapping pixels)
            max_count_row = (
                get_max_percent_row(df) if percent else get_max_count_row(df)
            )
            reference_group_list.append(max_count_row["reference_group"])
            comparison_group_list.append(max_count_row["comparison_group"])

            # drop_already_assigned_values
            df = drop_already_assigned_values(
                df, reference_group_list + comparison_group_list
            )

    remap_dict = calculate_remap_dict(reference_group_list, comparison_group_list)

    return remap_dict


#  --------------------------
#  Region Algorithm
#  --------------------------


def count_mappings_for_regions(
    reference_da: xr.DataArray, comparison_da: xr.DataArray, regions: List[Region]
) -> pd.DataFrame:
    all_df = []
    variable = reference_da.name

    for region in regions:
        # add error catching in case need to invert latlon
        try:
            region_reference_da = select_bounding_box(
                reference_da.to_dataset(), region
            )[variable]
        except AssertionError:
            region_reference_da = select_bounding_box(
                reference_da.to_dataset(), region, inverse_lat=True
            )[variable]
        try:
            region_comparison_da = select_bounding_box(
                comparison_da.to_dataset(), region
            )[variable]
        except AssertionError:
            region_comparison_da = select_bounding_box(
                comparison_da.to_dataset(), region, inverse_lat=True
            )[variable]

        # count the pixels in each group
        d = convert_counts_dict_to_dataframe(
            count_matching_pixels(region_reference_da, region_comparison_da)
        )
        d["region"] = [region.name for _ in range(len(d))]
        all_df.append(d)

    all_df = pd.concat(all_df)

    return all_df


def match_by_region_algorithm(
    all_df: pd.DataFrame, regions: List[Region]
) -> Dict[float, float]:
    """ Use predefined Region objects to subset the spatial maps and
    run the maximum count algorithm to map the groupings for these specific
    areas across months!
    """
    assert all(
        np.isin(["reference_group", "comparison_group"], [c for c in all_df.columns])
    ), f"Need columns reference_group comparison_group. Found: {all_df.columns}"

    reference_group_list = []
    comparison_group_list = []
    n_clusters = len(all_df["comparison_group"].unique())

    for region in regions[:n_clusters]:
        region_name = region.name

        # IF only one group-value left, assign it to the final remaining group (itself)
        if len(all_df["comparison_group"].unique()) == 1:
            # final value is itself
            #  remap_dict[df['comparison_group'].unique()[0]] = df['comparison_group'].unique()[0]
            reference_group_list.append(all_df["reference_group"].unique()[0])
            comparison_group_list.append(all_df["comparison_group"].unique()[0])
            all_df = drop_already_assigned_values(
                all_df, reference_group_list + comparison_group_list
            )
        else:
            region_df = all_df.query(f"region == '{region_name}'")

            max_count_row = get_max_count_row(region_df)

            # save the mapping
            reference_group_list.append(max_count_row["reference_group"])
            comparison_group_list.append(max_count_row["comparison_group"])

            # drop_already_assigned_values
            all_df = all_df.loc[
                ~np.isin(all_df["comparison_group"], comparison_group_list)
            ]
            all_df = all_df.loc[
                ~np.isin(all_df["reference_group"], reference_group_list)
            ]

    remap_dict = calculate_remap_dict(reference_group_list, comparison_group_list)

    return remap_dict


# Plotting helper functions
def plot_colors_remapping(colors, remap_dict) -> None:
    colors_remapped = [[int(v) for v in remap_dict.values()]]
    sns.palplot(colors)
    sns.palplot(colors_remapped)


def plot_comparisons(
    reference_da: xr.DataArray,
    comparison_da: xr.DataArray,
    colors: List[str],
    remapping_dict: Union[Dict[int, int], Dict[float, float]],
    title: Optional[str] = None,
    num_ks: int = 5,
) -> None:
    """check that the remapping is sensible"""
    # convert to numpy array
    remapping_list = np.array([int(k) for k in remapping_dict.values()])

    # plot the colormaps
    sns.palplot(colors)
    ax = plt.gca()
    ax.set_xticks([i for i in range(0, n)])
    ax.set_xticklabels([i for i in range(0, n)])
    if title is not None:
        ax.set_title(title)
    sns.palplot(np.array(colors)[remapping_list])
    ax = plt.gca()
    ax.set_xticks([i for i in range(0, n)])
    ax.set_xticklabels(remapping_list)

    # create cmaps
    new_cmap = ListedColormap(colors[remapping_list])
    cmap = ListedColormap(colors)

    # plot the spatial patterns
    fig, axs = plt.subplots(1, 3, figsize=((6.4 / 2) * 3, 4.8))
    axs[1].imshow(reference_da.values[::-1, :], cmap=cmap)
    axs[1].set_title("Reference DA")
    axs[0].imshow(comparison_da.values[::-1, :], cmap=cmap)
    axs[0].set_title("Comparison DA")
    axs[2].imshow(comparison_da.values[::-1, :], cmap=new_cmap)
    axs[2].set_title("Comparison DA Remapped")


def sort_by_another_list(list_to_sort, list_to_sort_on):
    assert len(list_to_sort) == len(list_to_sort_on)
    sort_ixs = np.argsort(list_to_sort_on)
    return list_to_sort[sort_ixs]


def run_clustering(
    month_embeddings: np.ndarray,
    month_pred_months: np.ndarray,
    month_latlons: np.ndarray,
    ks: List[int] = [5],
) -> Tuple[xr.Dataset, Dict[int, KMeans]]:
    """for each unique static embedding (currently months - to capture seasonality,
    but could be 1D).

    Returns:
    One cluster_ds and one list of estimators (for different ks)
    """
    # calculate clusters for ALL x.nc inputs (each month)
    all_cluster_ds = []
    all_estimators = []

    for ix, (embedding, pred_month, latlons) in tqdm.tqdm(
        enumerate(zip(month_embeddings, month_pred_months, month_latlons))
    ):
        # fit the clusters
        static_clusters, estimators = fit_kmeans(embedding, ks)

        # convert to dataset
        pixels = latlons
        lons = latlons[:, 1]
        lats = latlons[:, 0]
        static_cluster_ds = convert_clusters_to_ds(
            ks, static_clusters, pixels, lats, lons, time=ix
        )

        # append to final list
        all_cluster_ds.append(static_cluster_ds)
        all_estimators.append(estimators)

    #  combine into one xr.Dataset
    cluster_ds = xr.auto_combine(all_cluster_ds)

    return cluster_ds, all_estimators


# ---------------------------------------------------------
# Get the region bounding boxes
# ---------------------------------------------------------
# def get_regions_for_clustering_boxes(ds: xr.Dataset) -> List[Region]:
#     """Because we defined the latlon boxes by their numerical
#     index we have to get the values for the latlon boxes by the
#     `.isel()` method on one of the preprocessed datasets.
#     """
#     kitui = Region(
#         name="kitui",
#         lonmin=ds.isel(lon=13).lon.values,
#         lonmax=ds.isel(lon=19).lon.values,
#         latmin=ds.isel(lat=-34).lat.values,
#         latmax=ds.isel(lat=-24).lat.values,
#     )
#     victoria = Region(
#         name="victoria",
#         lonmin=ds.isel(lon=0).lon.values,
#         lonmax=ds.isel(lon=12).lon.values,
#         latmin=ds.isel(lat=-31).lat.values,
#         latmax=ds.isel(lat=-15).lat.values,
#     )
#     turkana_edge = Region(
#         name="turkana_edge",
#         lonmin=ds.isel(lon=14).lon.values,
#         lonmax=ds.isel(lon=29).lon.values,
#         latmin=ds.isel(lat=-9).lat.values,
#         latmax=ds.isel(lat=-2).lat.values,
#     )
#     nw_pastoral = Region(
#         name="nw_pastoral",
#         lonmin=ds.isel(lon=0).lon.values,
#         lonmax=ds.isel(lon=12).lon.values,
#         latmin=ds.isel(lat=-6).lat.values,
#         latmax=ds.isel(lat=-1).lat.values,
#     )
#     coastal = Region(
#         name="coastal",
#         lonmin=ds.isel(lon=21).lon.values,
#         lonmax=ds.isel(lon=34).lon.values,
#         latmin=ds.isel(lat=-25).lat.values,
#         latmax=ds.isel(lat=-13).lat.values,
#     )

#     regions = [coastal, victoria, nw_pastoral, kitui, turkana_edge]

#     return regions


def get_regions_for_clustering_boxes(ds: xr.Dataset) -> List[Region]:
    """Because we defined the latlon boxes by their numerical
    index we have to get the values for the latlon boxes by the
    `.isel()` method on one of the preprocessed datasets.

    FOR THE NEW CLUSTERINGS
    """
    victoria = Region(
        name="victoria",
        lonmin=ds.isel(lon=0).lon.values,
        lonmax=ds.isel(lon=7).lon.values,
        latmin=ds.isel(lat=-28).lat.values,
        latmax=ds.isel(lat=-18).lat.values,
    )
    turkana = Region(
        name="turkana",
        lonmin=ds.isel(lon=5).lon.values,
        lonmax=ds.isel(lon=16).lon.values,
        latmin=ds.isel(lat=-16).lat.values,
        latmax=ds.isel(lat=-6).lat.values,
    )
    southern_highlands = Region(
        name="southern_highlands",
        lonmin=ds.isel(lon=3).lon.values,
        lonmax=ds.isel(lon=13).lon.values,
        latmin=ds.isel(lat=-41).lat.values,
        latmax=ds.isel(lat=-31).lat.values,
    )
    coastal = Region(
        name="coastal",
        lonmin=ds.isel(lon=15).lon.values,
        lonmax=ds.isel(lon=20).lon.values,
        latmin=ds.isel(lat=-44).lat.values,
        latmax=ds.isel(lat=-34).lat.values,
    )
    nw_pastoral = Region(
        name="nw_pastoral",
        lonmin=ds.isel(lon=0).lon.values,
        lonmax=ds.isel(lon=12).lon.values,
        latmin=ds.isel(lat=-6).lat.values,
        latmax=ds.isel(lat=-1).lat.values,
    )

    regions = [coastal, victoria, turkana, southern_highlands, nw_pastoral]

    return regions


# ---------------------------------------------------------
# Remap the values in DataArray
# ---------------------------------------------------------


def remap_values(da: xr.DataArray, transdict: Dict) -> xr.DataArray:
    vals = da.values
    new_vals = np.copy(vals)

    # replace values
    for k, v in transdict.items():
        new_vals[vals == k] = v

    return xr.ones_like(da) * new_vals


def remap_all_monthly_values(
    cluster_ds: xr.Dataset, remap_dicts: Dict[Union[str, int], Union[str, float]]
) -> xr.Dataset:
    """From the remap dictionaries, select the correct cluster variable
    from cluster_ds (the value for k) and remap all those values using the values
    seen in the remap_dict.
    """
    remapped_ds = cluster_ds.copy()
    assert len(remapped_ds.time) == 12, "Expected time to be size 12 (monthly)"

    k = len([v for v in remap_dicts["Feb"].values()])
    # for each month in cluster_ds
    all_remapped = []
    for time in range(1, 12):
        transdict = remap_dicts[calendar.month_abbr[time + 1]]

        all_remapped.append(
            remap_values(  # type: ignore
                da=remapped_ds[f"cluster_{k}"].isel(time=time), transdict=transdict
            )
        )

    # join each month back into one Dataset
    remapped_ds = xr.concat(
        [remapped_ds[f"cluster_{k}"].isel(time=0)] + all_remapped, dim="time"
    )
    remapped_ds = remapped_ds.to_dataset()
    return remapped_ds


if __name__ == "__main__":
    EXPERIMENT = "2020_04_28:143300_one_month_forecast_BASE_static_vars"
    TRUE_EXPERIMENT = "one_month_forecast"

    data_dir = get_data_path()

    # -------------------
    # 1. Load the experiment!
    ealstm = load_model(data_dir / "models" / EXPERIMENT / "ealstm" / "model.pt")
    ealstm.models_dir = data_dir / "models" / EXPERIMENT

    ealstm.experiment = TRUE_EXPERIMENT

    # -------------------
    #  2. Calculate the static embedding
    (
        all_e,
        (all_static_x, all_latlons, all_pred_months,),
    ) = get_static_embedding(  #  type: ignore
        ealstm=ealstm
    )
    pred_months_err_mask = [len(np.unique(pm)) == 1 for pm in all_pred_months]
    all_e = np.array(all_e)[pred_months_err_mask]  #  type: ignore
    all_static_x = np.array(all_static_x)[pred_months_err_mask]  #  type: ignore
    all_pred_months = np.array(all_pred_months)[pred_months_err_mask]  #  type: ignore
    all_latlons = np.array(all_latlons)[pred_months_err_mask]  #  type: ignore

    assert all_latlons.shape == all_static_x.shape  #  type: ignore
    assert all_pred_months.shape == all_e.shape  #  type: ignore

    # assert all timsteps have only 1 pred month
    assert all([i == 1 for i in [len(np.unique(pm)) for pm in all_pred_months]])

    # SORTBY month
    pred_months = [int(np.unique(pm)) for pm in all_pred_months]

    all_e = sort_by_another_list(all_e, pred_months)
    all_static_x = sort_by_another_list(all_static_x, pred_months)
    all_latlons = sort_by_another_list(all_latlons, pred_months)
    all_pred_months = sort_by_another_list(all_pred_months, pred_months)

    pred_months = [int(np.unique(pm)) for pm in all_pred_months]

    # Get only the static embeddings for each month
    unique_ids = [pred_months.index(x) for x in set(pred_months)]
    month_embeddings = all_e[unique_ids]  # type: ignore
    month_static_x = all_static_x[unique_ids]  # type: ignore
    month_latlons = all_latlons[unique_ids]  # type: ignore
    month_pred_months = all_pred_months[unique_ids]  # type: ignore
    month_pred_months = [np.unique(m)[0] for m in month_pred_months]

    # -------------------
    # 3. run the clustering
    ks = [5]

    cluster_ds = run_clustering(
        month_embeddings=month_embeddings,
        month_pred_months=month_pred_months,
        month_latlons=month_latlons,
        ks=ks,
    )
    # get the regions
    regions = get_regions_for_clustering_boxes(cluster_ds)

    # -------------------
    # 4. Get the matching groups and plot

    #                    yellow,    green,    turqoise,   blue,      purple
    colors = np.array(["#fde832", "#67c962", "#43928d", "#3b528b", "#461954"])
    cmap = ListedColormap(colors)

    # remap_dict, matches_df = get_matching_groups(reference_da, comparison_da)
    jan = cluster_ds.isel(time=0).cluster_5
    may = cluster_ds.isel(time=4).cluster_5

    comparison_da = may
    reference_da = jan

    remap_dict, df = get_matching_groups(reference_da, comparison_da, regions=regions)

    print("Remapping:", remap_dict.values())

    ## CHECK that the remapping is sensible!
    plot_comparisons(
        reference_da,
        comparison_da,
        colors=colors,
        remapping_dict=remap_dict,
        title=None,
    )

    # get each month remapping dictionary and plot
    fig, axs = plt.subplots(4, 3, figsize=(15, 8 * 3))
    colors = np.array(["#fde832", "#67c962", "#43928d", "#3b528b", "#461954"])
    cmap = ListedColormap(colors)
    reference_da.plot(ax=axs[0, 0], add_colorbar=False, cmap=cmap)
    axs[0, 0].set_title(calendar.month_abbr[1])

    remap_dicts = {}

    for mth in range(1, 12):
        ax = axs[np.unravel_index(mth, (4, 3))]
        comparison_da = cluster_ds.isel(time=mth).cluster_5
        remap_dict, matches_df = get_matching_groups(
            reference_da, comparison_da, regions=regions
        )
        new_cmap = ListedColormap(
            colors[np.array([int(i) for i in remap_dict.values()])]
        )
        comparison_da.plot(add_colorbar=False, ax=ax, cmap=new_cmap)
        ax.set_title(calendar.month_abbr[mth + 1])
        remap_dicts[calendar.month_abbr[mth + 1]] = remap_dict

    # -------------------
    # 4. Get the matching groups
    remapped_ds = remap_all_monthly_values(cluster_ds, remap_dicts)  # type: ignore

    remapped_ds.to_netcdf(data_dir / "tommy/cluster_ds.nc")

    # plot each month with same colormap
    fig, axs = plt.subplots(4, 3, figsize=(15, 8 * 3))

    cmap = ListedColormap(colors)

    for mth in range(0, 12):
        ax = axs[np.unravel_index(mth, (4, 3))]
        remapped_ds.cluster_5.isel(time=mth).plot(add_colorbar=False, ax=ax, cmap=cmap)
        ax.set_title(calendar.month_abbr[mth + 1])
