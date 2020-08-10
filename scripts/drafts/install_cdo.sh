# https://www.studytrails.com/blog/install-climate-data-operator-cdo-with-netcdf-grib2-and-hdf5-support/#comment-4094
# https://code.mpimet.mpg.de/projects/cdo/embedded/index.html#x1-30001.1
# https://www.unidata.ucar.edu/software/netcdf/docs/getting_and_building_netcdf.html
# https://gist.github.com/mainvoid007/e5f1c82f50eb0459a55dfc4a0953a08e

## zlib
cd ~
wget ftp://ftp.unidata.ucar.edu/pub/netcdf/netcdf-4/zlib-1.2.8.tar.gz
gunzip zlib-1.2.8.tar.gz
tar xf zlib-1.2.8.tar
cd ~/zlib-1.2.8/
# ZDIR=/usr/local
# ./configure --prefix=${ZDIR}
./configure –prefix=/opt/cdo-install
make
make check
sudo make install

## HDF5
cd ~
wget ftp://ftp.unidata.ucar.edu/pub/netcdf/netcdf-4/hdf5-1.8.13.tar.gz
gunzip hdf5-1.8.13.tar.gz
tar hdf5-1.8.13.tar
cd ~/hdf5-1.8.13/
./configure --with-zlib=/opt/cdo-install prefix=/opt/cdo-install CFLAGS=-fPIC
make
make check
sudo make install

## Netcdf4
cd ~
wget ftp://ftp.unidata.ucar.edu/pub/netcdf/netcdf-4.3.3.1.tar.gz
gunzip netcdf-4.3.3.1.tar.gz
tar xf netcdf-4.3.3cd.1.tar
cd ~/netcdf-4.3.3.1/
CPPFLAGS=-I/opt/cdo-install/include LDFLAGS=-L/opt/cdo-install/lib ./configure prefix=/opt/cdo-install CFLAGS=-fPIC
sudo make
make check
sudo make install

# ## JASPER
# cd ~


# ## GRIB
# cd ~
# wget https://confluence.ecmwf.int/download/attachments/45757960/eccodes-2.18.0-Source.tar.gz
# gunzip eccodes-2.18.0-Source.tar.gz
# tar xf eccodes-2.18.0-Source.tar
# cd ~/eccodes-2.18.0-Source/
# ./configure –prefix=/opt/cdo-install
# sudo make
# make check
# sudo make install

## CDO
cd ~
wget https://code.mpimet.mpg.de/attachments/download/20826/cdo-1.9.8.tar.gz

gunzip cdo-1.9.8.tar.gz
tar xf cdo-1.9.8.tar
cd ~/cdo-1.9.8
./configure --prefix=/opt/cdo-install CFLAGS=-fPIC --with-netcdf=/opt/cdo-install --with-hdf5=/opt/cdo-install

make
make check
sudo make install
