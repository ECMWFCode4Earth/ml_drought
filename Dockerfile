# FROM creates a layer from the continuumio/miniconda3 Docker image (`base image`)
# use the current official miniconda image as the basis for your image
FROM continuumio/miniconda3

# create a directory ml_drought in the container
RUN mkdir /ml_drought
WORKDIR /ml_drought

# COPY adds files from your Docker client’s current directory
# copy the environment.yml file to the docker container and build conda env
COPY environment.yml .
RUN conda env create -f ./environment.yml

# copy the code to the docker container
COPY src ./src
COPY tests ./tests
COPY *.ini ./
COPY pipeline_config ./pipeline_config
