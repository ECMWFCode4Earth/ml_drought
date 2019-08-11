from pathlib import Path
import xarray as xr
import pandas as pd
from datetime import datetime
from typing import Tuple, Dict, List, Union, Optional
import warnings
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class RegionAnalysis:
    """Create summary statistics for all Regions (defined as xr.Dataset objects)
    comparing the model predictions against the true values for both train and
    test datasets.

    This class produces two different formats for the same data. Firstly, it
    writes a csv file for each model-region object.
    Secondly, it saves all of the data to a class attribute (`self.df`) in long
    format. This allows the class to then produce summary statistics.

    Attributes:
    -----------
    :region_ds: xr.DataArray
    :region_lookup: Dict
    :self.pred_variable: str
    :self.true_variable: str
    :self.models_dir: Path
    :self.features_dir: Path
    :self.models: List[str]

    TODO:
    # train or test `true` data ?
    self.mode = 'test' if test_mode else 'train'

    Note:
    - because we only save data from the `test` data (vs. the `train` data)
     we can currently only use data from:
     `(data/features/{experiment}/test).glob('**/*.nc')` to do our analysis
    - this doesn't have to be the case if we use the trained model and push the
    data in `X.nc` through
    """
    df: pd.DataFrame

    def __init__(self,
                 data_dir: Path = Path('data'),
                 experiment: str = 'one_month_forecast',
                 admin_boundaries: bool = True):
        self.pred_variable: Optional[str] = None
        self.true_variable: Optional[str] = None

        self.models_dir: Path = data_dir / 'models' / experiment
        self.features_dir: Path = data_dir / 'features' / experiment / 'test'
        assert self.models_dir.exists(), 'Require {data_path}/models to have been'\
            'created by the pipeline.'
        assert self.features_dir.exists(), 'Require {data_path}/features to have been'\
            'created by the pipeline.'

        self.models: List[str] = [m.name for m in self.models_dir.iterdir()]

        assert experiment in ['one_month_forecast', 'nowcast']
        self.experiment: str = experiment

        # NOTE: this shouldn't be specific for the boundaries it should
        # also be able to work with landcover
        if admin_boundaries:
            self.shape_data_dir = data_dir / 'analysis' / 'boundaries_preprocessed'
        else:
            self.shape_data_dir = data_dir / 'interim' / 'static' / 'landcover'
        self.region_data_paths: List[Path] = [f for f in self.shape_data_dir.glob('*.nc')]

        self.out_dir: Path = data_dir / 'analysis' / 'region_analysis'
        if not self.out_dir.exists():
            self.out_dir.mkdir(parents=True, exist_ok=True)

        print(f'Initialised the Region Analysis for experiment: {self.experiment}')
        print(f'Models: {self.models}')
        print(f'Regions: {[r.name for r in self.region_data_paths]}')
        # print(f'Test timesteps: {}')

    def load_region_data(self, region_data_path: Path) -> Tuple[xr.DataArray, Dict, str]:
        # LOAD in region lookup DataArray
        assert 'analysis' in region_data_path.parts, 'Only preprocessed' \
            'region files (as netcdf) should be used' \
            '`data/analysis`'
        region_group_name: str = region_data_path.name
        region_ds: xr.Dataset = xr.open_dataset(region_data_path)
        region_da: xr.DataArray = region_ds[[v for v in region_ds.data_vars][0]]
        region_lookup: Dict = dict(zip(
            [int(k.strip()) for k in region_ds.attrs['keys'].split(',')],
            [v.strip() for v in region_ds.attrs['values'].split(',')]
        ))

        return region_da, region_lookup, region_group_name

    def load_prediction_data(self, preds_data_path: Path) -> xr.DataArray:
        assert 'models' in preds_data_path.parts, 'Only modelled' \
            'from the pipeline should be used using this class' \
            '`data/models`'

        pred_ds = xr.open_dataset(preds_data_path)
        # get the variable = Dataset -> DataArray
        if self.pred_variable is None:
            # Check that variables are only length 1
            pred_variables = [v for v in pred_ds.data_vars]
            assert len(pred_variables) == 1, 'Only expect one variable in pred_ds'
            self.pred_variable = pred_variables[0]

        return pred_ds[self.pred_variable]

    def load_true_data(self, true_data_path: Path) -> xr.DataArray:
        assert 'features' in true_data_path.parts, 'Only engineered data' \
            'from the pipeline should be used using this class' \
            '`data/features`'
        true_ds = xr.open_dataset(true_data_path)
        # Dataset -> DataArray
        if self.true_variable is None:
            # Check that variables are only length 1
            true_variables = [v for v in true_ds.data_vars]
            assert len(true_variables) == 1, 'Only expect one variable in true_ds'
            self.true_variable = true_variables[0]

        return true_ds[self.true_variable]

    def compute_mean_statistics(self, region_da: xr.DataArray,
                                region_lookup: Dict,
                                pred_da: xr.DataArray,
                                true_da: xr.DataArray,
                                datetime: datetime) -> Tuple[List, List, List, List]:
        # For each region calculate mean `target_variable` in true / pred
        valid_region_ids: List = [k for k in region_lookup.keys()]
        region_name: List = []
        predicted_mean_value: List = []
        true_mean_value: List = []
        datetimes: List = []

        for valid_region_id in valid_region_ids:
            region_name.append(region_lookup[valid_region_id])
            predicted_mean_value.append(
                pred_da.where(region_da == valid_region_id).mean().values
            )
            true_mean_value.append(
                true_da.where(region_da == valid_region_id).mean().values
            )
            # assert true_da.time == pred_da.time, 'time must be matching!'
            datetimes.append(datetime)

        assert len(region_name) == len(predicted_mean_value) == len(datetimes)
        return datetimes, region_name, predicted_mean_value, true_mean_value

    def get_pred_data_on_timestep(self, datetime: datetime, model: str) -> Path:
        # TODO: fix this method to be more flexible to higher time-resolution data
        warnings.warn('This functionality only works with MONTHLY predictions')
        month = int(datetime.month)
        year = int(datetime.year)

        preds_data_path = (
            self.models_dir / model / f'preds_{year}_{month}.nc'
        )
        return preds_data_path

    @staticmethod
    def read_xr_datetime(xr_obj: Union[xr.Dataset, xr.DataArray]) -> datetime:
        dt = pd.to_datetime(xr_obj.time.values)
        assert len(dt) == 1, 'only meant to have ONE datetime in this example'

        return dt.to_pydatetime()[0]

    @staticmethod
    def get_all_timesteps(path_to_dir: Path) -> List[datetime]:
        # TODO: this should be dynamically selecting timesteps from xr objects
        # not manually creating strings that are in the format of interest
        # defined by the filenames/folder names
        # >>> get_all_timesteps(self.features_dir)
        # >>> get_all_timesteps(self.models_dir / 'ealstm')
        return [datetime(1, 1, 1)]

    def _analyze_single_shapefile(self, region_data_path: Path) -> Union[pd.DataFrame, None]:
        admin_level_name = region_data_path.name.replace('.nc', '')
        # for ONE REGION compute the each model statistics (pred & true)
        region_da, region_lookup, region_group_name = self.load_region_data(region_data_path)

        for model in self.models:
            print(f'\n** Calculating for {model} **')
            # create the filename
            if not (self.out_dir / model).exists():
                (self.out_dir / model).mkdir(exist_ok=True, parents=True)

            # TODO: rewrite this once the preds data has timestamps
            #      (less fiddly with getting associated data for preds/true)
            # for each timestep in the test data get associated PREDICTED data
            dfs = []
            true_data_paths = [f for f in self.features_dir.glob('*/y.nc')]  # if f.name == 'y.nc'
            for true_data_path in true_data_paths:
                # load the required data
                true_da = self.load_true_data(true_data_path)
                dt = self.read_xr_datetime(true_da)
                preds_data_path = self.get_pred_data_on_timestep(
                    datetime=dt, model=model
                )
                if not preds_data_path.exists():
                    # if it's not there
                    warnings.warn(
                        f'{preds_data_path.parents[0] / preds_data_path.name} does not exist'
                    )
                    continue
                pred_da = self.load_prediction_data(preds_data_path)
                # compute the statistics
                (
                    datetimes, region_name, predicted_mean_value, true_mean_value
                ) = self.compute_mean_statistics(
                    region_da=region_da, true_da=true_da, pred_da=pred_da,
                    region_lookup=region_lookup, datetime=dt
                )
                # store as pandas object and add to
                dfs.append(pd.DataFrame({
                    'admin_level_name': [admin_level_name for _ in range(len(datetimes))],
                    'model': [model for _ in range(len(datetimes))],
                    'datetime': datetimes,
                    'region_name': region_name,
                    'predicted_mean_value': predicted_mean_value,
                    'true_mean_value': true_mean_value,
                }))

            if dfs == []:
                print(f'No matching time data found for {model}')
                print(f'Contents of {model} dir:')
                print(f'\t{[f.name for f in (self.models_dir / model).iterdir()]}')
                return

            else:
                df = pd.concat(dfs).reset_index()
                df = df.sort_values(by=['datetime'])
                output_filepath = self.out_dir / model / f'{model}_{admin_level_name}.csv'
                df.to_csv(output_filepath)
                print(f'** Written output csv to {output_filepath.as_posix()} **')
                return df

    def compute_global_error_metrics(self) -> None:
        models = []
        rmses = []
        maes = []
        r2s = []
        for model in self.df.model.unique():
            mean_model_performance = (
                self.df
                .loc[self.df.model == model][['predicted_mean_value', 'true_mean_value']]
            )
            rmse = np.sqrt(mean_squared_error(
                mean_model_performance.true_mean_value,
                mean_model_performance.predicted_mean_value
            ))
            mae = mean_absolute_error(
                mean_model_performance.true_mean_value,
                mean_model_performance.predicted_mean_value
            )
            r2 = r2_score(
                mean_model_performance.true_mean_value,
                mean_model_performance.predicted_mean_value
            )
            models.append(model)
            rmses.append(rmse)
            maes.append(mae)
            r2s.append(r2)

        self.global_mean_metrics = pd.DataFrame({
            'model': models,
            'rmse': rmses,
            'mae': maes,
            'r2': r2s,
        })
        print('* Assigned Global Error Metrics to `self.global_mean_metrics` *')

    def analyze(self) -> None:
        """For all preprocessed regions"""
        all_regions_dfs = []
        for region_data_path in self.region_data_paths:
            df = self._analyze_single_shapefile(region_data_path)
            all_regions_dfs.append(df)

        # clean up the DataFrame
        all_regions_df = pd.concat(all_regions_dfs).reset_index()
        if 'index' in all_regions_df.columns:
            all_regions_df = all_regions_df.drop(columns='index')
        if 'level_0' in all_regions_df.columns:
            all_regions_df = all_regions_df.drop(columns='level_0')

        self.df = all_regions_df
        print('* Assigned all region dfs to `self.df` *')

        # compute error metrics for each model globally
        self.compute_global_error_metrics()
