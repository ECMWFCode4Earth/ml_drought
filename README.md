<!-- ![](https://imgur.com/8qjbXcD) -->
[![Build Status](https://travis-ci.com/esowc/ml_drought.svg?branch=master)](https://travis-ci.com/esowc/ml_drought)

<!-- https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb#scrollTo=8QAWNjizy_3O -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/esowc/ml_drought/blob/master/notebooks/docs/Pipeline.ipynb)

<!-- OVERVIEW -->

# A Machine Learning Pipeline for Climate Science

This repository is an end-to-end pipeline for the creation, intercomparison and evaluation of machine learning methods in climate science.

The pipeline carries out a number of tasks to create a unified-data format for training and testing machine learning methods.

These tasks are split into the different classes defined in the `src` folder and explained further below:

<img src="https://github.com/esowc/ml_drought/blob/master/img/pipeline_overview.png" width="600"  style="margin:auto; width:70%; padding:10px;">

NOTE: some basic working knowledge of Python is required to use this pipeline, although it is not too onerous

<!-- HOW TO USE THE PIPELINE -->

## Using the Pipeline <a name="using"></a>

There are three entrypoints to the pipeline:
* [run.py](run.py)
* [notebooks](notebooks/docs)
* [scripts](scripts/README.md)

A blog post describing the goals and design of the pipeline can be found
[here](https://medium.com/@gabrieltseng/a-machine-learning-pipeline-for-climate-research-ebf83b2b349a).

View the initial presentation of our pipeline [here](https://www.youtube.com/watch?v=QVFiGERCiYs).

## Setup <a name="setup"></a>

[Anaconda](https://www.anaconda.com/download/#macos) running python 3.7 is used as the package manager. To get set up
with an environment, install Anaconda from the link above, and (from this directory) run

```bash
conda env create -f environment.yml
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
See [here](https://stackoverflow.com/a/33596140/9940782) for help on all machines or [here](https://stackoverflow.com/a/49719638/9940782)
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

You will also need to create a .cdsapirc file with the following information:
```bash
url: https://cds.climate.copernicus.eu/api/v2
key: <INSERT KEY HERE>
verify: 1
```

### Testing  <a name="testing"></a>

This pipeline can be tested by running `pytest`. [flake8](http://flake8.pycqa.org) is used for linting.

We use [mypy](https://github.com/python/mypy) for type checking. This can be run by running `mypy src` (this runs mypy on the `src` directory).

We use [black](https://black.readthedocs.io/en/stable/) for code formatting.

<!-- PROJECT TEAM MEMBERS -->

__Team:__ [@tommylees112](https://github.com/tommylees112), [@gabrieltseng](https://github.com/gabrieltseng)

For updates follow [@tommylees112](https://twitter.com/tommylees112) on twitter or look out for our blog posts!

- [Blog 1: Great News!](https://tommylees112.github.io/posts/2019/1/esowc_kick_off)
- [Blog 2: The Pipeline](https://medium.com/@gabrieltseng/a-machine-learning-pipeline-for-climate-research-ebf83b2b349a)
- [Blog 3: The Close of the Project!](https://tommylees112.github.io/posts/2019/2/esowc_final)

<!-- ESoWC initial outline -->

## Acknowledgements <a name="acknowledgements"></a>
This was a project completed as part of the ECMWF Summer of Weather Code [Challenge #12](https://github.com/esowc/challenges_2019/issues/14). The challenge was setup to use [ECMWF/Copernicus open datasets](https://cds.climate.copernicus.eu/#!/home) to evaluate machine learning techniques for the **prediction of droughts**.

Huge thanks to @ECMWF for making this project possible!
