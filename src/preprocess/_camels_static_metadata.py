import numpy as np
import xarray as xr
from typing import Dict, List


groupings: Dict = {
    "climatic_attributes": [
        "gauge_id",
        "p_mean",
        "pet_mean",
        "aridity",
        "p_seasonality",
        "frac_snow",
        "high_prec_freq",
        "high_prec_dur",
        "high_prec_timing",
        "low_prec_freq",
        "low_prec_dur",
        "low_prec_timing",
    ],
    "humaninfluence_attributes": [
        "gauge_id",
        "benchmark_catch",
        "surfacewater_abs",
        "groundwater_abs",
        "discharges",
        "abs_agriculture_perc",
        "abs_amenities_perc",
        "abs_energy_perc",
        "abs_environmental_perc",
        "abs_industry_perc",
        "abs_watersupply_perc",
        "num_reservoir",
        "reservoir_cap",
        "reservoir_he",
        "reservoir_nav",
        "reservoir_drain",
        "reservoir_wr",
        "reservoir_fs",
        "reservoir_env",
        "reservoir_nousedata",
        "reservoir_year_first",
        "reservoir_year_last",
    ],
    "hydrogeology_attributes": [
        "gauge_id",
        "inter_high_perc",
        "inter_mod_perc",
        "inter_low_perc",
        "frac_high_perc",
        "frac_mod_perc",
        "frac_low_perc",
        "no_gw_perc",
        "low_nsig_perc",
        "nsig_low_perc",
    ],
    "hydrologic_attributes": [
        "gauge_id",
        "q_mean",
        "runoff_ratio",
        "stream_elas",
        "slope_fdc",
        "baseflow_index",
        "baseflow_index_ceh",
        "hfd_mean",
        "Q5",
        "Q95",
        "high_q_freq",
        "high_q_dur",
        "low_q_freq",
        "low_q_dur",
        "zero_q_freq",
    ],
    "hydrometry_attributes": [
        "gauge_id",
        "station_type",
        "flow_period_start",
        "flow_period_end",
        "flow_perc_complete",
        "bankfull_flow",
        "structurefull_flow",
        "q5_uncert_upper",
        "q5_uncert_lower",
        "q25_uncert_upper",
        "q25_uncert_lower",
        "q50_uncert_upper",
        "q50_uncert_lower",
        "q75_uncert_upper",
        "q75_uncert_lower",
        "q95_uncert_upper",
        "q95_uncert_lower",
        "q99_uncert_upper",
        "q99_uncert_lower",
        "quncert_meta",
    ],
    "landcover_attributes": [
        "gauge_id",
        "dwood_perc",
        "ewood_perc",
        "grass_perc",
        "shrub_perc",
        "crop_perc",
        "urban_perc",
        "inwater_perc",
        "bares_perc",
        "dom_land_cover",
    ],
    "soil_attributes": [
        "gauge_id",
        "sand_perc",
        "sand_perc_missing",
        "silt_perc",
        "silt_perc_missing",
        "clay_perc",
        "clay_perc_missing",
        "organic_perc",
        "organic_perc_missing",
        "bulkdens",
        "bulkdens_missing",
        "bulkdens_5",
        "bulkdens_50",
        "bulkdens_95",
        "tawc",
        "tawc_missing",
        "tawc_5",
        "tawc_50",
        "tawc_95",
        "porosity_cosby",
        "porosity_cosby_missing",
        "porosity_cosby_5",
        "porosity_cosby_50",
        "porosity_cosby_95",
        "porosity_hypres",
        "porosity_hypres_missing",
        "porosity_hypres_5",
        "porosity_hypres_50",
        "porosity_hypres_95",
        "conductivity_cosby",
        "conductivity_cosby_missing",
        "conductivity_cosby_5",
        "conductivity_cosby_50",
        "conductivity_cosby_95",
        "conductivity_hypres",
        "conductivity_hypres_missing",
        "conductivity_hypres_5",
        "conductivity_hypres_50",
        "conductivity_hypres_95",
        "root_depth",
        "root_depth_missing",
        "root_depth_5",
        "root_depth_50",
        "root_depth_95",
        "soil_depth_pelletier",
        "soil_depth_pelletier_missing",
        "soil_depth_pelletier_5",
        "soil_depth_pelletier_50",
        "soil_depth_pelletier_95",
    ],
    "topographic_attributes": [
        "gauge_id",
        "gauge_name",
        "gauge_lat",
        "gauge_lon",
        "gauge_easting",
        "gauge_northing",
        "gauge_elev",
        "area",
        "dpsbar",
        "elev_mean",
        "elev_min",
        "elev_10",
        "elev_50",
        "elev_90",
        "elev_max",
    ],
}

# helper functions
def remove_from_grouping(group: str, vars_: List[str]) -> Tuple[List[str], List[str]]:
    """
    e.g.
    > vars_ = ['aridity', 'frac_snow',]
    > remove_from_grouping('climat', vars_)
    """
    vars_ += ["gauge_id"]
    keys = np.array([k for k in groupings.keys()])
    group_key = keys[[group in k for k in keys]][0]

    leftover = [
        v
        for v in groupings[group_key]
        if v not in [var_to_remove for var_to_remove in vars_]
    ]
    print(f"removing {vars_} from {group_key}")
    return leftover, vars_


def check_correctly_spelt(static_ds: xr.Dataset, var_list: List[str]) -> None:
    """
    e.g.
    check_correctly_spelt(static_ds, vars_to_drop)
    """
    # are they all spelled correctly?
    correctly_spelt = np.isin(var_list, list(static_ds.data_vars))
    if not all(correctly_spelt):
        print(np.array(var_list)[~correctly_spelt])


# functions for exploring the static data
def overview_dict(ds: xr.Dataset, vars_: List[str], limit: int = 5) -> Dict:
    """Overview information from static_ds variables"""
    return {
        v: (
            f"n_unique: {len(np.unique(ds[v].values))}",
            f"dtype: {ds[v].dtype}",
            f"#nulls: {ds[v].isnull().sum().values}",
            np.unique(ds[v].values)[:limit],
        )
        for v in vars_
    }


def get_var_on_dtype(ds, dtype: str, limit: int = 5) -> Dict:
    """return overview of of variables of dtype"""
    vars_ = [v for v in ds.data_vars if ds[v].dtype == np.dtype(dtype)]
    return overview_dict(ds, vars_, limit=limit)


def get_vars_on_keywords(ds, keywords: List[str], limit: int = 5) -> Dict:
    """return overview of variables containing keyword"""
    if isinstance(keywords, str):
        keywords = [keywords]
    vars_ = [v for v in ds.data_vars if any([keyword in v for keyword in keywords])]
    return overview_dict(ds, vars_, limit=limit)


def get_vars_on_grouping(ds, group: str, limit: int = 5) -> Dict:
    """return overview of variables in group"""
    keys = np.array([k for k in groupings.keys()])
    group_key = keys[[group in k for k in keys]][0]

    vars_ = ["gauge_id"]
    leftover = [
        v
        for v in groupings[group_key]
        if v not in [var_to_remove for var_to_remove in vars_]
    ]

    valid_vars = list(ds.data_vars)
    return overview_dict(ds, [v for v in leftover if v in valid_vars], limit=limit)


# variables to ignore
static_ignore_vars = [
    # hydrometry_attributes
    "station_type",
    "flow_period_start",
    "flow_period_end",
    "flow_perc_complete",
    "bankfull_flow",
    "structurefull_flow",
    "q5_uncert_upper",
    "q5_uncert_lower",
    "q25_uncert_upper",
    "q25_uncert_lower",
    "q50_uncert_upper",
    "q50_uncert_lower",
    "q75_uncert_upper",
    "q75_uncert_lower",
    "q95_uncert_upper",
    "q95_uncert_lower",
    "q99_uncert_upper",
    "q99_uncert_lower",
    "quncert_meta",
    # soil_attributes
    "sand_perc",
    "sand_perc_missing",
    "silt_perc",
    "silt_perc_missing",
    "clay_perc",
    "clay_perc_missing",
    "organic_perc",
    "organic_perc_missing",
    "bulkdens",
    "bulkdens_missing",
    "bulkdens_5",
    "bulkdens_50",
    "bulkdens_95",
    "tawc",
    "tawc_missing",
    "tawc_5",
    "tawc_50",
    "tawc_95",
    "porosity_cosby",
    "porosity_cosby_missing",
    "porosity_cosby_5",
    "porosity_cosby_50",
    "porosity_cosby_95",
    "porosity_hypres_missing",
    "porosity_hypres_5",
    "porosity_hypres_50",
    "porosity_hypres_95",
    "conductivity_cosby",
    "conductivity_cosby_missing",
    "conductivity_cosby_5",
    "conductivity_cosby_50",
    "conductivity_cosby_95",
    "conductivity_hypres_missing",
    "conductivity_hypres_5",
    "conductivity_hypres_50",
    "conductivity_hypres_95",
    "root_depth",
    "root_depth_missing",
    "root_depth_5",
    "root_depth_50",
    "root_depth_95",
    "soil_depth_pelletier",
    "soil_depth_pelletier_missing",
    "soil_depth_pelletier_5",
    "soil_depth_pelletier_50",
    "soil_depth_pelletier_95",
    # landcover_attributes
    "grass_perc",
    "shrub_perc",
    "inwater_perc",
    "bares_perc",
    "dom_land_cover",
    # topographic_attributes
    "gauge_name",
    "gauge_lat",
    "gauge_lon",
    "gauge_easting",
    "gauge_northing",
    "gauge_elev",
    "area",
    "elev_mean",
    "elev_min",
    "elev_10",
    "elev_50",
    "elev_90",
    "elev_max",
    # hydrologic_attributes (?? I like these ??)
    "q_mean",
    "runoff_ratio",
    "stream_elas",
    "slope_fdc",
    "baseflow_index",
    "baseflow_index_ceh",
    "hfd_mean",
    "Q5",
    "Q95",
    "high_q_freq",
    "high_q_dur",
    "low_q_freq",
    "low_q_dur",
    "zero_q_freq",
    # hydrogeology_attributes
    "inter_high_perc",
    "inter_mod_perc",
    "inter_low_perc",
    "frac_high_perc",
    "frac_mod_perc",
    "frac_low_perc",
    "no_gw_perc",
    "low_nsig_perc",
    "nsig_low_perc",
    # humaninfluence_attributes
    "benchmark_catch",
    "discharges",
    "abs_agriculture_perc",
    "abs_amenities_perc",
    "abs_energy_perc",
    "abs_environmental_perc",
    "abs_industry_perc",
    "abs_watersupply_perc",
    "num_reservoir",
    "reservoir_he",
    "reservoir_nav",
    "reservoir_drain",
    "reservoir_wr",
    "reservoir_fs",
    "reservoir_env",
    "reservoir_nousedata",
    "reservoir_year_first",
    "reservoir_year_last",
    # climatic_attributes
    "p_mean",
    "pet_mean",
    "p_seasonality",
    "high_prec_freq",
    "high_prec_dur",
    "high_prec_timing",
    "low_prec_freq",
    "low_prec_dur",
    "low_prec_timing",
]
