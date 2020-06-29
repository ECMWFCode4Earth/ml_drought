# /soge-home/users/chri4118/.conda/envs/ml/bin/pip
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# exec $SHELL
# git clone https://github.com/tommylees112/tommy_multiple_forcing
# conda activate ml
# export PATH=/opt/conda/envs/ml/bin:$PATH
# conda install pytorch torchvision cudatoolkit=10.1 -c pytorch  --yes

conda install -c conda-forge xarray=0.15.1 --yes
conda install -c conda-forge shap==0.30 --yes
conda install h5py --yes
conda install torch --yes
conda install ipython --yes
conda install pip --yes
conda install -c conda-forge xesmf --yes
conda install matplotlib --yes
conda install numba --yes

pip install netcdf4
pip install pytorch-lightning
pip install geopandas
pip install sklearn
pip install mypy
pip install pytest
pip install ruamel.yaml