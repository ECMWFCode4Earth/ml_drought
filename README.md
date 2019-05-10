# ml_drought [![Build Status](https://travis-ci.com/esowc/ml_drought.svg?branch=master)](https://travis-ci.com/esowc/ml_drought)

## ESoWC 2019 - Machine learning to better predict and understand drought.

Using [ECMWF/Copernicus open datasets](https://cds.climate.copernicus.eu/#!/home) to evaluate machine learning techniques for the **prediction of droughts**.

<!-- PROJECT SHIELDS -->

<!-- PROJECT LOGO -->

<!-- <br />
<p align="center">
  <a href="https://github.com/esowc/ml_drought">
    <img src="logo.png" alt="Logo" width="80" height="80">
  </a>
</p> -->

<!-- PROJECT TEAM MEMBERS -->

__Team:__ [@tommylees112](https://github.com/tommylees112), [@gabrieltseng](https://github.com/gabrieltseng)

<!-- TODO List -->
## To do
- [x] Write an initial README.md
- [ ] Choose datasets that initially interested in
- [ ] Figure out how best to interact with Copernicus (API, web portal, other)


<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
* [Work in Progress](#work-in-progress)
* [Setup](#setup)
* [Acknowledgements](#acknowledgements)

## About the Project <a name="about-the-project"></a>
> The Summer of Weather Code(ESoWC) programme by the European Centre for Medium-Range Weather Forecasts (ECMWF) is a collabrative online programme to promote the development of weather-related open-source software.

This is our contribution to the ECMWF Summer of Weather Code programme where we will be developing a pipeline for rapid experimentation of:
- different machine learning algorithms
- different input datasets (feature selection)
- different measures of drought (meteorological, hydrological, agricultural)

Our goals are as follows:
- To build a robust, useful pipeline.
- Usefully predict drought.
- Understand the relationships between model inputs and outputs - *What is the model learning?*
- Make the pipeline and results accessible.

## Work in progress <a name="work-in-progress"></a>

We will be documenting the progress of our pipeline as go.

For updates follow [@tommylees112](https://twitter.com/tommylees112) on twitter or look out for our blog posts!

- [Blog 1: Great News!](https://tommylees112.github.io/posts/2019/1/esowc_kick_off)

## Setup <a name="setup"></a>

[Anaconda](https://www.anaconda.com/download/#macos) running python 3.7 is used as the package manager. To get set up
with an environment, install Anaconda from the link above, and (from this directory) run

```bash
conda env create -f environment.{mac, ubuntu.cpu}.yml
```
This will create an environment named `esowc-drought` with all the necessary packages to run the code. To
activate this environment, run

```bash
conda activate esowc-drought
```

[Docker](https://www.docker.com/) can also be used to run this code. To do this, first
run the docker app (either [docker desktop](https://www.docker.com/products/docker-desktop))
or configure the `docker-machine`:

```bash
# on macOS
brew install docker-machine docker

docker-machine create --driver virtualbox default
docker-machine env default
```
See [here](https://stackoverflow.com/a/33596140/9940782) for help or [here](https://stackoverflow.com/a/49719638/9940782)
for MacOS.


Then build the docker image:

```bash
docker build -t ml_drought .
```

Then, use it to run a container, mounting the data folder to the container:

```bash
docker run -it \
--mount type=bind,source=<PATH_TO_DATA>,target=/ml_drought/data \
ml_drought /bin/bash
```

This pipeline can be tested by running `pytest`.

## Acknowledgements <a name="acknowledgements"></a>
Huge thanks to @ECMWF for making this project possible!
