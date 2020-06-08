"""
Create a dataframe of the variables in the Adede et al paper (2019)

In [239]: all_df.columns
Out[239]:
Index(['datetime', 'region_name', 'RCI1M', 'RCI3M', 'RFE1M', 'RFE3M', 'SPI1',
       'SPI3', 'VCI1M', 'VCI3M', 'RCI1M_t1', 'RCI3M_t1', 'RFE1M_t1',
       'RFE3M_t1', 'SPI1_t1', 'SPI3_t1', 'VCI1M_t1', 'VCI3M_t1', 'RCI1M_t2',
       'RCI3M_t2', 'RFE1M_t2', 'RFE3M_t2', 'SPI1_t2', 'SPI3_t2', 'VCI1M_t2',
       'VCI3M_t2', 'RCI1M_t3', 'RCI3M_t3', 'RFE1M_t3', 'RFE3M_t3', 'SPI1_t3',
       'SPI3_t3', 'VCI1M_t3', 'VCI3M_t3'],
      dtype='object')
"""

import pickle
from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

data_dir = Path("/Volumes/Lees_Extend/data/ecmwf_sowc/data")
# data_dir = Path('data')

from src.engineer import Engineer

# # e = Engineer(data_dir)
# data = e.engineer_class._make_dataset(static=False)

from src.analysis import read_train_data, read_test_data
from src.analysis.indices.utils import rolling_mean


boku = True

if boku:
    experiment = "one_month_forecast_BOKU_boku_VCI"
else:
    experiment = "one_month_forecast"  # "one_month_forecast_BOKU_boku_VCI"

X_train, y_train = read_train_data(data_dir, experiment=experiment)
X_test, y_test = read_test_data(data_dir, experiment=experiment)
ds = xr.merge([y_train, y_test]).sortby("time").sortby("lat")
d_ = xr.merge([X_train, X_test]).sortby("time").sortby("lat")
ds = xr.merge([ds, d_])


# ----------------------------------------
# Create the features (pixel-by-pixel)
# ----------------------------------------

"""
NOTE: Nasty hack

the indices.spi computation sometimes collapses the dimensionality
of the groupby object

~/miniconda3/envs/crop/lib/python3.7/site-packages/xarray/core/computation.py in apply_variable_ufunc(func, signature, exclude_dims, dask, output_dtypes, output_sizes, keep_attrs, *args)
    580             data = np.expand_dims(data, -1)
    581
--> 582         if data.ndim != len(dims):
    583             raise ValueError(
    584                 'applied function returned data with unexpected '

ValueError: applied function returned data with unexpected number of
dimensions: 1 vs 2, for dimensions ('time', 'point')

ipdb> dims
('time', 'point')
ipdb> data.shape
(464,)
ipdb> data = np.expand_dims(data, -1)
ipdb> data.shape
(464, 1)

Therefore added the lines (L578: ~/miniconda3/envs/crop/lib/python3.7/site-packages/xarray/core/computation.py):
>        # TODO: TOMMY ADDED
>        if (data.ndim == 1) and (len(dims) == 2):
>            data = np.expand_dims(data, -1)
>

"""
# create SPI data
from src.analysis import SPI

spi = SPI(ds=ds[["precip"]])
spi.fit(variable="precip", scale=1, calibration_year_final=2010)
SPI1M = spi.index
spi.fit(variable="precip", scale=3, calibration_year_final=2010)
SPI3M = spi.index

# create RCI data
from src.analysis.indices import ConditionIndex

rci = ConditionIndex(ds=ds[["precip"]], resample_str="M")
rci.fit(variable="precip", rolling_window=1)
var_ = [v for v in rci.index.data_vars][0]
RCI1M = rci.index.rename({var_: "RCI1M"})
rci.fit(variable="precip", rolling_window=3)
var_ = [v for v in rci.index.data_vars][0]
RCI3M = rci.index.rename({var_: "RCI3M"})

# create month aggregations VCI / precip
if boku:
    vci_variable = "boku_VCI"  # "VCI"
else:
    vci_variable = "VCI"

vci = ds[[vci_variable]].resample(time="M").mean(dim="time")
VCI1M = rolling_mean(vci, 1).rename({vci_variable: "VCI1M"})
VCI3M = rolling_mean(vci, 3).rename({vci_variable: "VCI3M"})

precip = ds[["precip"]].resample(time="M").mean(dim="time")
RFE1M = rolling_mean(precip, 1).rename({"precip": "RFE1M"})
RFE3M = rolling_mean(precip, 3).rename({"precip": "RFE3M"})

# make into one dataframe
out_ds = xr.auto_combine([VCI1M, VCI3M, RFE1M, RFE3M, SPI1M, SPI3M, RCI1M, RCI3M])
# select the legitimate timesteps (before all missing data)
from_ = out_ds.isel(time=2).time
to_ = f"{out_ds.isel(time=-1)['time.year'].values + 1}-01-01"
out_ds = out_ds.sel(time=slice(from_, to_))

# ---------------
# Save to disk
# ---------------

if boku:
    out_dir = data_dir / "interim" / "BOKU_adede_preprocessed"
else:
    out_dir = data_dir / "interim" / "adede_preprocessed"

if not (out_dir).exists():
    (out_dir).mkdir(parents=True, exist_ok=True)
out_ds.to_netcdf(out_dir / "data_kenya.nc")

# ----------------------------------------
# Convert to a geopandas dataframe
# ----------------------------------------
# make into region geodataframe
from src.analysis import AdministrativeRegionAnalysis
from src.analysis.region_analysis import KenyaGroupbyRegion

# GROUP results by region data
grouper = KenyaGroupbyRegion(data_dir)

# just run for the first region to get the output geodataframe
gdf = grouper.analyze(da=out_ds[[v for v in out_ds.data_vars][0]], selection="level-2")
gdf_lookup = gdf.groupby("region_name").first()
gdf_lookup = gdf_lookup.reset_index().drop(columns="mean_value")

# ---------------
# Save to disk
# ---------------
import pickle

out_dir = data_dir / "interim" / "adede_preprocessed"
if not (out_dir).exists():
    (out_dir).mkdir(parents=True, exist_ok=True)

with open(out_dir / "region_gdf_lookup.pkl", "wb") as fp:
    pickle.dump(gdf_lookup, fp)


# all_gdfs = []
# for var_ in [v for v in out_ds.data_vars]:
#     print(f"\n** Working on variable: {var_} **")
#     gdf = grouper.analyze(da=out_ds[var_], selection='level-2')
#     gdf = gdf.rename(columns={'mean_value': 'SPI3'})
#     all_gdfs.append(gdf)
#     print(f"** Done variable: {var_} **\n")

# ----------------------------------------
# Convert to a pandas dataframe
# ----------------------------------------

# ALTERNATIVE:
# 1) load the shapefile into an xr.DataArray object
# 2) Mask and mean each region over time
# 3) return a PANDAS DATAFRAME instead of GEODATAFRAME
# 4) calculate a geodataframe just for REFERENCE / to lookup

grouper = KenyaGroupbyRegion(data_dir)

region_shp_path = (
    data_dir / "analysis" / "boundaries_preprocessed" / "district_l2_kenya.nc"
)
admin_level_name = "district_l2"
region_da, region_lookup, region_group_name = grouper.load_region_data(region_shp_path)
var_0 = [v for v in out_ds.data_vars][0]
df = grouper.calculate_mean_per_region(
    da=out_ds[var_0],
    region_da=region_da,
    region_lookup=region_lookup,
    admin_level_name=admin_level_name,
)
df = df.rename(columns={"mean_value": var_0})

all_dfs = []
for var_ in [v for v in out_ds.data_vars][1:]:
    print(f"\n** Working on variable: {var_} **")
    d = grouper.calculate_mean_per_region(
        da=out_ds[var_],
        region_da=region_da,
        region_lookup=region_lookup,
        admin_level_name=admin_level_name,
    )
    d = d.rename(columns={"mean_value": var_})
    all_dfs.append(d)
    print(f"** Done variable: {var_} **\n")

# vars_ = [v for v in out_ds.data_vars][1:]
# all_dfs = [d.rename(columns={'mean_value': var_}) for d in all_dfs for var_ in vars_]
# df_ = df.copy()

# merge into ONE dataframe
for i in range(len(all_dfs)):
    try:
        df = df.merge(all_dfs[i])
    except TypeError as E:
        print(E)
        continue


# ---------------
# Save to disk
# ---------------
import pickle

out_dir = data_dir / "interim" / "adede_preprocessed"
if not (out_dir).exists():
    (out_dir).mkdir(parents=True, exist_ok=True)

with open(out_dir / "adede_variables_df.pkl", "wb") as fp:
    pickle.dump(df, fp)

df.to_csv(out_dir / "adede_variables_df.csv")

# ----------------------------------------
# calculate the lagged variables
# (SO THAT each row has all features)
# ----------------------------------------

variable_columns = [c for c in df.columns if c not in ["datetime", "region_name"]]

all_lagged_dfs = []
for lag in [1, 2, 3]:
    new_colnames = [f"{cname}_t{lag}" for cname in variable_columns]
    column_rename_map = dict(zip(variable_columns, new_colnames))

    all_lagged_dfs.append(
        df.groupby("region_name")[variable_columns]
        .shift(lag)
        .rename(columns=column_rename_map)
    )

all_df = pd.concat([df] + all_lagged_dfs, axis=1)

# ---------------
# Save to disk
# ---------------

out_dir = data_dir / "interim" / "adede_preprocessed"
if not (out_dir).exists():
    (out_dir).mkdir(parents=True, exist_ok=True)

with open(out_dir / "ALL_adede_variables_df.pkl", "wb") as fp:
    pickle.dump(all_df, fp)

all_df.to_csv(out_dir / "ALL_adede_variables_df.csv")
