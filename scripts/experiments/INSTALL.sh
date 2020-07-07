# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# bash Miniconda3-latest-Linux-x86_64.sh
# exec $SHELL
# /soge-home/users/chri4118/.conda/envs/ml/bin/pip
conda install -c conda-forge xarray=0.15.1 --yes
conda install -c conda-forge shap==0.30 --yes
conda install h5py --yes
conda install torch --yes
conda install ipython --yes
conda install pip --yes
conda install -c conda-forge xesmf --yes

pip install netcdf4
pip install pytorch-lightning
pip install geopandas