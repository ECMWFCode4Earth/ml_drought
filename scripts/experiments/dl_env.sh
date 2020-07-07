conda create -n ml
conda activate ml
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
pip install ruamel.yaml
git clone https://github.com/esowc/ml_drought.git
git clone https://github.com/tommylees112/tommy_multiple_forcing.git
conda install pytest
conda install ipython --yes
conda install pandas --yes
conda install xarray --yes
conda install tqdm --yes
conda install numba --yes
conda install h5py --yes
conda install matplotlib --yes
conda install scipy --yes
pip install TensorBoard