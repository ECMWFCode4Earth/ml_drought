#!/bin/bash

#   Institut für Wetter- und Klimakommunikation GmbH / Qmet
#   O. Maywald <maywald@klimagipfel.de>

#   This should install CDO with grib2, netcdf and HDF5 support. Note that the binaries are in /opt/cdo-install/bin.
#   For further information look:
#   http://www.studytrails.com/blog/install-climate-data-operator-cdo-with-netcdf-grib2-and-hdf5-support/

#   docker-command
#   use this command to start a docker container
#    docker run -it --name cdo --rm -v $(pwd):/opt/cdo-install -w /opt/cdo-install ubuntu:latest bash

home=/opt/cdo-install
apt-get update && apt-get install -y wget build-essential checkinstall unzip m4 curl libcurl4-gnutls-dev

#   download, compile and install --> zlib
cd $home
wget ftp://ftp.unidata.ucar.edu/pub/netcdf/netcdf-4/zlib-1.2.8.tar.gz
tar -xzvf zlib-1.2.8.tar.gz
cd zlib-1.2.8
./configure -prefix=/opt/cdo-install
make && make check && make install

#   download, compile and install --> hdf5
cd $home
wget ftp://ftp.unidata.ucar.edu/pub/netcdf/netcdf-4/hdf5-1.8.13.tar.gz
tar -xzvf hdf5-1.8.13.tar.gz
cd hdf5-1.8.13
./configure -with-zlib=/opt/cdo-install -prefix=/opt/cdo-install CFLAGS=-fPIC
make && make check && make install

#   download, compile and install --> netCDF
cd $home
wget ftp://ftp.unidata.ucar.edu/pub/netcdf/netcdf-4.5.0.tar.gz
tar -xzvf netcdf-4.5.0.tar.gz
cd netcdf-4.5.0/
CPPFLAGS=-I/opt/cdo-install/include LDFLAGS=-L/opt/cdo-install/lib ./configure -prefix=/opt/cdo-install CFLAGS=-fPIC
make && make check && make install

#   download, compile and install --> jasper
cd $home
wget http://www.ece.uvic.ca/~frodo/jasper/software/jasper-1.900.1.zip
unzip jasper-1.900.1.zip
cd jasper-1.900.1
./configure -prefix=/opt/cdo-install CFLAGS=-fPIC
make && make check && make install

#   download, compile and install --> grib_api
# cd $home
# wget https://software.ecmwf.int/wiki/download/attachments/3473437/grib_api-1.24.0-Source.tar.gz?api=v2 -O grib_api-1.24.0.tar.gz
# tar -xzvf grib_api-1.24.0.tar.gz
# cd grib_api-1.24.0-Source
# ./configure -prefix=/opt/cdo-install CFLAGS=-fPIC -with-netcdf=/opt/cdo-install -with-jasper=/opt/cdo-install
# make && make check && make install

#   download, compile and install --> cdo
cd $home
wget https://code.mpimet.mpg.de/attachments/download/15653/cdo-1.9.1.tar.gz
tar -xvzf cdo-1.9.1.tar.gz
cd cdo-1.9.1
./configure -prefix=/opt/cdo-install CFLAGS=-fPIC -with-netcdf=/opt/cdo-install -with-jasper=/opt/cdo-install -with-hdf5=/opt/cdo-install   # -with-grib_api=/opt/cdo-install/grib_api-1.24.0-Source
make && make check && make install

#   set PATH
PATH=/opt/cdo-install/bin:$PATH
#echo "PATH=\"/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/opt/cdo-install/bin\"" > /etc/environment
