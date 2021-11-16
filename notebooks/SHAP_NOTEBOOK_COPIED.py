import sys
sys.path.append("/home/leest/ml_drought")
from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

data_dir = Path('/DataDrive200/data/')

import seaborn as sns
from src.utils import drop_nans_and_flatten

from src.analysis import read_train_data, read_test_data, read_pred_data

EXPERIMENT =      'one_month_forecast'
TRUE_EXPERIMENT = 'one_month_forecast'
TARGET_VAR =      'boku_VCI'
# TARGET_VAR =      'VCI3M'

assert (data_dir / f'models/{EXPERIMENT}').exists()
assert (data_dir / f'models/{TRUE_EXPERIMENT}').exists()

print((data_dir / f'models/{EXPERIMENT}').as_posix())

X_train, y_train = read_train_data(data_dir, experiment=TRUE_EXPERIMENT)
X_test, y_test = read_test_data(data_dir, experiment=TRUE_EXPERIMENT)
static_ds = xr.open_dataset(data_dir / "features/static/data.nc")

ds = xr.merge([y_train, y_test]).sortby('time').sortby('lat')
d_ = xr.merge([X_train, X_test]).sortby('time').sortby('lat')
ds = xr.merge([ds, d_])

# Mask the data
from src.utils import get_ds_mask
mask = get_ds_mask(X_train.VCI)

ealstm_pred = read_pred_data('ealstm', data_dir, experiment=EXPERIMENT)[-1].where(~mask)

fig, ax = plt.subplots()
time_ix = 0
time = ealstm_pred.isel(time=time_ix).time.values
ealstm_pred.isel(time=time_ix).plot(ax=ax)
ax.set_title(f"** PREDICTED VALUES {str(time).split('T')[0]}**", size=16);

from src.models import load_model

ealstm = load_model(data_dir / 'models' / EXPERIMENT / 'ealstm' / 'model.pt')
ealstm.models_dir = data_dir / 'models' / EXPERIMENT

ealstm.experiment = TRUE_EXPERIMENT

# Exmaine hte data 
# ealstm_ignore_vars = [v_ for v_ in ealstm.ignore_vars if v_ != "t2m"]

dynamic_ds = ds.drop([v for v in ealstm.ignore_vars if v in list(ds.data_vars)])

print(dynamic_ds.data_vars)
ealstm.features_per_month  # x.shape[-1] = the number of features in dynamic data 

# Check the static data 
dl = ealstm.get_dataloader('train', batch_file_size=1, shuffle_data=False)
train_iter = iter(dl)
static_data = train_iter.static
print(f"N Vars: {len(list(static_data.data_vars))}")
static_data

# Run the Shap Analysis
"""
- Run the `scripts/experiments/22_shap_analysis.py` file
- `ipython --pdb 22_shap_analysis.py`
"""

from pandas.tseries.offsets import MonthEnd
from typing import Dict, List
from collections import namedtuple


def get_timestep_from_date_str(date_str) -> pd.Timestamp:
    return pd.to_datetime(date_str, format="%Y_%m") + MonthEnd()


def open_shap_analysis(model) -> Dict[str, namedtuple]:  # type: ignore
    """Read the data from the SHAP analysis run in the other functions"""
    ShapValues = namedtuple(
        "ShapValues", ["date_str", "target_time", "historical", "pred_month", "static"]
    )

    analysis_dir = model.model_dir / "analysis"
    dirs = [d for d in analysis_dir.iterdir() if len(list(d.glob("*.nc"))) > 0]

    out_dict = {}
    for shap_analysis_dir in dirs:
        shap = ShapValues(
            date_str=shap_analysis_dir.name,
            target_time=get_timestep_from_date_str(shap_analysis_dir.name),
            historical=xr.open_dataset(shap_analysis_dir / "shap_historical_ds.nc"),
            pred_month=xr.open_dataset(shap_analysis_dir / "shap_pred_month.nc"),
            static=xr.open_dataset(shap_analysis_dir / "shap_static.nc"),
        )
        out_dict[shap_analysis_dir.name] = shap

    return out_dict

shap = open_shap_analysis(ealstm)
shap.keys()

shap["2016_11"].historical

# 1. add target time dimension
shap_t = shap['2018_2']
all_shap = []
for shap_key in ([k for k in shap.keys()]):
    shap_t = shap[shap_key]
    all_shap.append(
        shap_t.historical.expand_dims({'target_time': [shap_t.target_time]})
    )

# 2. merge on the new target_time dimension
all_shap = xr.concat(all_shap, dim='target_time')

# Get only the SHAP values for the previous time
# 1. add target time dimension
shap_t = shap['2018_2']
all_shap_t3 = []
for shap_key in ([k for k in shap.keys()]):
    shap_t = shap[shap_key]
    all_shap_t3.append(
        shap_t.historical.isel(time=-1).expand_dims({'target_time': [shap_t.target_time]})
    )

# 2. merge on the new target_time dimension
all_shap_t3 = xr.concat(all_shap_t3, dim='target_time')

# Get static data SHAP values
all_static = []
for shap_key in ([k for k in shap.keys()]):
    shap_t = shap[shap_key]
    all_static.append(
        shap_t.static.expand_dims({'target_time': [shap_t.target_time]})
    )

# 2. merge on the new target_time dimension
all_static = xr.concat(all_static, dim='target_time')

from src.utils import create_shape_aligned_climatology

# create pixel-mean and pixel_std
means = X_train.mean(dim='time').drop([v for v in ealstm.ignore_vars if v in list(X_train.data_vars)])
stds = X_train.std(dim='time').drop([v for v in ealstm.ignore_vars if v in list(X_train.data_vars)])
ones = xr.ones_like(X_test.drop([v for v in ealstm.ignore_vars if v in list(X_train.data_vars)]))
variable='precip'

means = ones*means
stds = ones*stds
# create_shape_aligned_climatology(X_train, means, variable, time_period='month')

# Get the X Test data (from the input data)
x_test = X_test.copy()
x_test = x_test.drop([v for v in ealstm.ignore_vars if v in list(x_test.data_vars)])
norm_x_test = (x_test - means) / stds
norm_x_test

norm_ds = ds.drop([v for v in ealstm.ignore_vars if v in list(ds.data_vars)])


# normalize the X data to compare with the shap values
# NEED TO NORMALIZE RELATIVE TO THE PIXEL MEANS ...
list_ds = []
for var in list(x_test.data_vars):
    list_ds.append((
        norm_ds[var] - ealstm.normalizing_dict[var]['mean']
    ) / ealstm.normalizing_dict[var]['std'])
    
    
norm_ds = xr.merge(list_ds)

# get teh normalised preds
# set the equivalent times
preds_mean = means[TARGET_VAR].isel(time=slice(0, len(ealstm_pred.time)))
preds_mean['time'] = ealstm_pred.time
preds_std = stds[TARGET_VAR].isel(time=slice(0, len(ealstm_pred.time)))
preds_std['time'] = ealstm_pred.time


norm_pred = (ealstm_pred - preds_mean) / preds_std
norm_pred = norm_pred.compute()
norm_pred = ((ealstm_pred.compute() - ealstm.normalizing_dict['boku_VCI']['mean']) / ealstm.normalizing_dict['boku_VCI']['std'])

# norm_pred

# Explore shap values 
shap["2016_5"].historical

def plot_shap_obs_pairs(shap_ds: xr.Dataset, obs_ds: xr.Dataset, variables: List[str], shap_plot_kwargs: Dict, obs_plot_kwargs: Dict, norm: bool = False, scale=1) -> None:
    n_variables = len(variables)
    assert np.unique(shap_ds.time.values).shape == (1,), "Only works for a single time"
    fig, axs = plt.subplots(n_variables, 2, figsize=((6*2)*scale, (4*n_variables)*scale))
    
    for ix, variable in enumerate(variables):
        if n_variables > 1:
            ax_row = axs[ix, :]
        else:
            ax_row = axs
        time = pd.to_datetime(
            shap_ds.time.values
        )

        # plot shap values
        shap_ds[variable].plot(ax=ax_row[0], **shap_plot_kwargs)
        ax_row[0].set_title(f"{time._date_repr}: SHAP {variable}")

        # plot the observed data
        obs_ds.sel(time=time)[variable].where(~mask).plot(ax=ax_row[1], **obs_plot_kwargs)
        title = "NORM" if norm else "RAW"
        ax_row[1].set_title(f"{time._date_repr}: {title} {variable}")
        
        for ax in ax_row:
            ax.axis('off')


variables = list(ds.drop([v for v in ealstm.ignore_vars if v in list(ds.data_vars)]).data_vars)

TIME = '2016_5'

plot_shap_obs_pairs(
    shap[TIME].historical.isel(time=-1), 
    norm_x_test, 
    ["precip"], 
    {'vmin': -0.4, 'vmax': 0.4, 'cmap': 'PiYG'}, 
    {'cmap': 'RdYlGn', 'vmin':-2, 'vmax':2}, 
    norm=True,
    scale=1.5
)
fig = plt.gcf()

for ax in fig.get_axes():
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]
              + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(14)


plot_shap_obs_pairs(
    shap[TIME].historical.isel(time=-1), 
    norm_x_test, 
    ["e"], 
    {'vmin': -0.4, 'vmax': 0.4, 'cmap': 'PiYG'}, 
    {'cmap': 'RdYlGn', 'vmin':-2, 'vmax':2}, 
    norm=True,
    scale=1.5
)
fig = plt.gcf()

for ax in fig.get_axes():
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]
              + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(14)

# Show predicted vs. target
from pandas.tseries.offsets import MonthEnd, DateOffset
target_time = pd.to_datetime(TIME, format="%Y_%m") + MonthEnd() + DateOffset(months=1)
target = norm_ds[TARGET_VAR].sel(time=target_time)
predicted = norm_pred

kwargs = {'cmap': 'RdYlGn', 'vmin':-2, 'vmax':2}
scale = 1.4
fig, ax = plt.subplots(1, 1, figsize=((6)*scale, (4)*scale))
target.plot(ax=ax, **kwargs)
ax.axis(False)
ax.set_title(f"Target {TARGET_VAR}: {pd.to_datetime(target.time.values)._date_repr}")
for ax in fig.get_axes():
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]
              + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(14)


predicted = norm_pred.sel(time=target_time)

kwargs = {'cmap': 'RdYlGn', 'vmin':-2, 'vmax':2}
scale = 1.4
fig, ax = plt.subplots(1, 1, figsize=((6)*scale, (4)*scale))
predicted.plot(ax=ax, **kwargs)
ax.axis(False)
ax.set_title(f"Predicted {TARGET_VAR}: {pd.to_datetime(target.time.values)._date_repr}")
for ax in fig.get_axes():
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]
              + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(14)


