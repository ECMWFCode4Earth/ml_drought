# conda draft
conda create -n tf --yes python=3.7
conda activate tf
# conda install pytorch==1.4.0 torchvision==0.5.0 -c pytorch --yes
conda install tensorflow -c anaconda --yes
conda install pytorch torchvision -c pytorch --yes
conda install -c conda-forge seaborn=0.11 --yes
conda install -c conda-forge netcdf4 numba tqdm jupyterlab tensorboard ipython pip ruamel.yaml xarray descartes statsmodels scikit-learn black mypy --yes
pip install geopandas

# conda xesmf
conda create -n xesmf python=3.7 --yes
conda activate xesmf
conda install -c conda-forge xesmf --yes
conda install dask netcdf4 ipython jupyterlab tqdm descartes black scikit-learn pandas pip ruamel.yaml -c conda-forge --yes
conda install pytorch -c pytorch --yes

# conda ml
conda create -n ml --yes 
conda activate ml
conda install -c conda-forge seaborn=0.11 --yes
conda install pytorch torchvision -c pytorch --yes
conda install -c conda-forge netcdf4 numba tqdm jupyterlab tensorboard ipython pip ruamel.yaml xarray descartes statsmodels scikit-learn black mypy --yes
pip install geopandas


# gdal env
conda create -n gdal --yes python=3.7
conda activate gdal
conda install -c conda-forge rasterio geopandas --yes
conda install -c anaconda tensorflow --yes
conda install -c conda-forge numpy ipython pip descartes scikit-learn --yes