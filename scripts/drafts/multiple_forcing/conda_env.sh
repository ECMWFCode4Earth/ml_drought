conda create -n ml --yes 
conda activate ml
conda install -c conda-forge seaborn=0.11 --yes
conda install pytorch torchvision -c pytorch --yes
conda install -c conda-forge netcdf4 numba tqdm jupyterlab tensorboard ipython pip ruamel.yaml xarray descartes statsmodels scikit-learn black mypy --yes
pip install geopandas
