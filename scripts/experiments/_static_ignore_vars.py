## variables to ignore
static_ignore_vars = [
    ## hydrometry_attributes -------------------
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
    ## soil_attributes -------------------
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
    ## landcover_attributes -------------------
    "grass_perc",
    "shrub_perc",
    "inwater_perc",
    "bares_perc",
    "dom_land_cover",
    ## topographic_attributes -------------------
    "gauge_name",
    "gauge_lat",
    "gauge_lon",
    "gauge_easting",
    "gauge_northing",
    "gauge_elev",
    "area",
    # "elev_mean",
    "elev_min",
    "elev_10",
    "elev_50",
    "elev_90",
    "elev_max",
    ## hydrologic_attributes (?? I like these ??) -------------------
    # "q_mean",
    "runoff_ratio",
    "stream_elas",
    # "slope_fdc",
    # "baseflow_index",
    "baseflow_index_ceh",
    "hfd_mean",
    "Q5",
    # "Q95",
    "high_q_freq",
    "high_q_dur",
    "low_q_freq",
    "low_q_dur",
    "zero_q_freq",
    ## hydrogeology_attributes -------------------
    "inter_high_perc",
    "inter_mod_perc",
    "inter_low_perc",
    "frac_high_perc",
    "frac_mod_perc",
    "frac_low_perc",
    "no_gw_perc",
    "low_nsig_perc",
    "nsig_low_perc",
    ## humaninfluence_attributes -------------------
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
    "surfacewater_abs",
    "groundwater_abs",
    ## climatic_attributes -------------------
    # "p_mean",
    "pet_mean",
    # "p_seasonality",
    # "high_prec_freq",
    "high_prec_dur",
    "high_prec_timing",
    "low_prec_freq",
    "low_prec_dur",
    "low_prec_timing",
]
