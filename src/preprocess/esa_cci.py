from pathlib import Path
import xarray as xr
import pandas as pd
from typing import Optional, List, Tuple
from shutil import rmtree
import numpy as np

from .base import BasePreProcessor
from ..utils import get_modal_value_across_time


class ESACCIPreprocessor(BasePreProcessor):
    """ Preprocesses the ESA CCI Landcover data """

    dataset = 'esa_cci_landcover'
    static = True

    @staticmethod
    def create_filename(netcdf_filepath: str,
                        subset_name: Optional[str] = None) -> str:
        """
        ESACCI-LC-L4-LCCS-Map-300m-P1Y-1992-v2.0.7b.nc
            =>
        ESACCI-LC-L4-LCCS-Map-300m-P1Y-1992-v2.0.7b_kenya.nc
        """
        if netcdf_filepath[-3:] == '.nc':
            filename_stem = netcdf_filepath[:-3]
        else:
            filename_stem = netcdf_filepath

        year = filename_stem.split('-')[-2]

        if subset_name is not None:
            new_filename = f"{year}_{filename_stem}_{subset_name}.nc"
        else:
            new_filename = f"{year}_{filename_stem}.nc"
        return new_filename

    @staticmethod
    def _map_to_groups(df: pd.DataFrame) -> pd.DataFrame:
        """ Reduce the number of landcover classes by grouping them together """
        map_ = {
            'No data': 'no_data',
            'Cropland, rainfed': 'cropland_rainfed',
            'Herbaceous cover': 'herbaceous_cover',
            'Tree or shrub cover': 'tree_or_shrub_cover',
            'Cropland, irrigated or post-flooding': 'cropland_irrigated_or_postflooding',
            'Mosaic cropland (>50%) / natural vegetation (tree, shrub, herbaceous cover) (<50%)': 'cropland_rainfed',
            'Mosaic natural vegetation (tree, shrub, herbaceous cover) (>50%) / cropland (<50%) ': 'tree_or_shrub_cover',
            'Tree cover, broadleaved, evergreen, closed to open (>15%)': 'tree_cover',
            'Tree cover, broadleaved, deciduous, closed to open (>15%)': 'tree_cover',
            'Tree cover, broadleaved, deciduous, closed (>40%)': 'tree_cover',
            'Tree cover, broadleaved, deciduous, open (15-40%)': 'tree_cover',
            'Tree cover, needleleaved, evergreen, closed to open (>15%)': 'tree_cover',
            'Tree cover, needleleaved, evergreen, closed (>40%)': 'tree_cover',
            'Tree cover, needleleaved, evergreen, open (15-40%)': 'tree_cover',
            'Tree cover, needleleaved, deciduous, closed to open (>15%)': 'tree_cover',
            'Tree cover, needleleaved, deciduous, closed (>40%)': 'tree_cover',
            'Tree cover, needleleaved, deciduous, open (15-40%)': 'tree_cover',
            'Tree cover, mixed leaf type (broadleaved and needleleaved)': 'tree_cover',
            'Mosaic tree and shrub (>50%) / herbaceous cover (<50%)': 'tree_or_shrub_cover',
            'Mosaic herbaceous cover (>50%) / tree and shrub (<50%)': 'herbaceous_cover',
            'Shrubland': 'shrubland',
            'Shrubland evergreen': 'shrubland',
            'Shrubland deciduous': 'shrubland',
            'Grassland': 'grassland',
            'Lichens and mosses': 'lichens_and_mosses',
            'Sparse vegetation (tree, shrub, herbaceous cover) (<15%)': 'tree_or_shrub_cover',
            'Sparse tree (<15%)': 'tree_cover',
            'Sparse shrub (<15%)': 'shrubland',
            'Sparse herbaceous cover (<15%)': 'herbaceous_cover',
            'Tree cover, flooded, fresh or brakish water': 'tree_cover',
            'Tree cover, flooded, saline water': 'tree_cover',
            'Shrub or herbaceous cover, flooded, fresh/saline/brakish water': 'shrubland',
            'Urban areas': 'urban_areas',
            'Bare areas': 'bare_areas',
            'Consolidated bare areas': 'bare_areas',
            'Unconsolidated bare areas': 'bare_areas',
            'Water bodies': 'water_bodies',
            'Permanent snow and ice': 'permanent_snow_and_ice',
        }
        # map to the groups defined above
        df['group_label'] = df.label.map(map_)

        # create ids
        keys = df['group_label'].unique()
        values = [i for i in range(len(keys))]
        df['group_value'] = df.group_label.map(dict(zip(keys, values)))

        return df

    @staticmethod
    def _reassign_lc_class_to_groups(ds: xr.Dataset, legend: pd.DataFrame) -> xr.Dataset:
        """ https://github.com/pydata/xarray/issues/2568#issuecomment-441343812 """
        print("Creating new DataArray with group values instead of original code")

        def remap(array, mapping):
            return np.array([mapping[k] for k in array.ravel()]).reshape(array.shape)

        mapping = dict(mapping=dict(
            zip(list(legend.code), list(legend.group_value))))
        ds['lc_class_group'] = xr.apply_ufunc(remap, ds, kwargs=mapping).lc_class

        return ds

    def _one_hot_encode(self, ds: xr.Dataset, group: bool = True) -> Tuple[xr.Dataset, pd.DataFrame]:

        legend = pd.read_csv(self.raw_folder / self.dataset / "legend.csv")
        # no data should have a value of 0 in the legend
        legend = self._map_to_groups(legend)

        if group:
            for grouped_val in legend.group_value.unique():
                group_values = list(
                    legend.loc[legend.group_value == grouped_val].code)
                group_label = str(
                    legend.loc[legend.group_value == grouped_val].group_label.unique()[
                        0]
                )
                ds[f"{group_label}_one_hot"] = ds.lc_class.where(
                    np.isin(ds.lc_class, group_values), 0
                ).clip(min=0, max=1)

        else:
            for idx, row in legend.iterrows():
                value, label = row.code, row.group_label
                ds[f"{label}_one_hot"] = ds.lc_class.where(ds.lc_class == value, 0).clip(
                    min=0, max=1
                )

        return ds, legend

    def _preprocess_single(self, netcdf_filepath: Path,
                           subset_str: Optional[str] = 'kenya',
                           regrid: Optional[xr.Dataset] = None) -> None:
        """ Preprocess a single netcdf file (run in parallel if
        `parallel_processes` arg > 1)

        Process:
        -------
        * chop region of interset (ROI)
        * regrid to same spatial grid as a reference dataset (`regrid`)
        * create new dataset with these dimensions
        * assign time stamp
        * Save the output file to new folder
        """
        assert netcdf_filepath.name[-3:] == '.nc', \
            f'filepath name should be a .nc file. Currently: {netcdf_filepath.name}'

        print(f'Starting work on {netcdf_filepath.name}')
        ds = xr.open_dataset(netcdf_filepath)

        # 2. chop out EastAfrica
        if subset_str is not None:
            try:
                ds = self.chop_roi(ds, subset_str)
            except AssertionError:
                print("Trying regrid again with inverted latitude")
                ds = self.chop_roi(ds, subset_str, inverse_lat=True)

        # 3. regrid to same spatial resolution ...?
        # NOTE: have to remove the extra vars for the regridder
        ds = ds.drop([
            'processed_flag', 'current_pixel_state',
            'observation_count', 'change_count', 'crs'
        ])
        if regrid is not None:
            ds = self.regrid(ds, regrid)

        try:  # try inferring from the ds.attrs
            time = pd.to_datetime(ds.attrs['time_coverage_start'])
        except KeyError:  # else infer from filename (for tests)
            year = netcdf_filepath.name.split('-')[-2]
            time = pd.to_datetime(f'{year}-01-01')

        ds = ds.assign_coords(time=time)
        ds = ds.expand_dims('time')

        # 5. extract the landcover data (reduce storage use)
        ds = ds.lccs_class.to_dataset(name='lc_class')

        # save to specific filename
        filename = self.create_filename(
            netcdf_filepath.name,
            subset_name=subset_str if subset_str is not None else None
        )
        print(f"Saving to {self.interim}/{filename}")
        ds.to_netcdf(self.interim / filename)

        print(f"** Done for ESA CCI landcover: {filename} **")

    def _preprocess_interim_files(self, subset_str: Optional[str] = 'kenya',
                                  one_hot_encode: bool = True,
                                  group: bool = True) -> None:
        ds = xr.open_mfdataset(self.get_filepaths('interim'))

        ds = get_modal_value_across_time(ds.lc_class).to_dataset()

        if one_hot_encode:
            ds, legend_df = self._one_hot_encode(ds, group=group)

        filename = self.dataset
        if subset_str is not None:
            filename = f'{filename}{"_" + subset_str}'
        if one_hot_encode:
            filename = f'{filename}_one_hot'

        filename = f'{filename}.nc'
        out = self.out_dir / filename

        if one_hot_encode:
            # write the output class (non-OHE map)
            lc_class_ds = self._reassign_lc_class_to_groups(
                ds.lc_class.to_dataset(name='lc_class'),
                legend_df
            )
            lc_class_ds.to_netcdf(self.out_dir / 'lc_class.nc')
            assert False
            legend_df.to_csv(self.out_dir / 'legend.csv')

        # write the OHE data (if used as static variables)
        ds = ds.drop("lc_class")
        ds.to_netcdf(out)

    def preprocess(self, subset_str: Optional[str] = 'kenya',
                   regrid: Optional[Path] = None,
                   years: Optional[List[int]] = None,
                   cleanup: bool = True,
                   one_hot_encode: bool = True,
                   group: bool = True) -> None:
        """Preprocess all of the ESA CCI landcover .nc files to produce
        one subset file resampled to the timestep of interest.
        (downloaded as annual timesteps)

        Arguments:
        ---------
        subset_str: Optional[str] = 'kenya'
            Whether to subset Kenya when preprocessing
        regrid: Optional[Path] = None
            If a Path is passed, the CHIRPS files will be regridded to have the same
            grid as the dataset at that Path. If None, no regridding happens
        years: Optional[List[int]] = None
            preprocess a subset of the years from the raw data
        cleanup: bool = True
            If true, delete interim files created by the class
        one_hot_encode: bool = True
            Whether to one hot encode the values
        group: bool = True
            Whether to group the landcover values
            (defined by the `map_` dict object in `_map_to_groups()`)

        Note:
        ----
        - because the landcover data only goes back to 1993 for all dates
        before 1993 that we need data for  we have selected the `modal`
        class from the whole data range (1993-2019).
        - This assumes that landcover is relatively consistent in the 1980s
        as the 1990s, 2000s and 2010s
        """
        print(f'Reading data from {self.raw_folder}. Writing to {self.interim}')

        nc_files = self.get_filepaths()
        if years is not None:
            nc_files = [f for f in nc_files if int(
                str(f).split('-')[-2]) in years
            ]

        if regrid is not None:
            regrid = self.load_reference_grid(regrid)

        for file in nc_files:
            self._preprocess_single(file, subset_str, regrid)

        self._preprocess_interim_files(
            subset_str=subset_str,
            one_hot_encode=one_hot_encode,
            group=group,
        )

        if cleanup:
            rmtree(self.interim)
