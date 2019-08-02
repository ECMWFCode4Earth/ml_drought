# ml_drought [![Build Status](https://travis-ci.com/esowc/ml_drought.svg?branch=master)](https://travis-ci.com/esowc/ml_drought)

Check out our companion repo with exploratory notebooks [here](https://github.com/tommylees112/esowc_notes)!

## ESoWC 2019 - Machine learning to better predict and understand drought.

Using [ECMWF/Copernicus open datasets](https://cds.climate.copernicus.eu/#!/home) to evaluate machine learning techniques for the **prediction of droughts**. This was a project proposal submitted to the ECMWF Summer of Weather Code [Challenge #12](https://github.com/esowc/challenges_2019/issues/14).

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

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
* [Work in Progress](#work-in-progress)
* [Pipeline](#pipeline)
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

We have a **set of notebooks and scripts** that are very rough but represent the current state of our work [at this repo here](https://github.com/tommylees112/esowc_notes).

For updates follow [@tommylees112](https://twitter.com/tommylees112) on twitter or look out for our blog posts!

- [Blog 1: Great News!](https://tommylees112.github.io/posts/2019/1/esowc_kick_off)
- [Blog 2: The Pipeline](https://medium.com/@gabrieltseng/a-machine-learning-pipeline-for-climate-research-ebf83b2b349a)

## Pipeline <a name="pipeline"></a>

A blog post describing the goals and design of the pipeline can be found 
[here](https://medium.com/@gabrieltseng/a-machine-learning-pipeline-for-climate-research-ebf83b2b349a).

Currently, the entrypoints into the pipeline are the scripts in the [scripts folder](scripts) - see the
[scripts README](scripts/README.md) for more information.

In the future, this will be replaced by the [`run.py`](run.py) file, with [json configurations](pipeline_config).
<!---
NOTE: RUN.PY AS DESCRIBED BELOW IS NOT FULLY IMPLEMENTED. IT IS COMMENTED OUT UNTIL THAT IS DONE. FOR NOW, THE ENTRYPOINT
TO THE PIPELINE WILL BE THE SCRIPTS IN THE SCRIPTS FOLDER.

The main entrypoint into the pipeline is [run.py](run.py). The configuration of the pipeline can be defined using a
[configuration file](pipeline_config) - the desired configuration file can be passed as a command line argument:

```bash
python run.py --config <PATH_TO_CONFIG>
```

If no configuration file is passed, the pipeline's [default minimal configuration](pipeline_config/minimal.json) is used.
-->

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

This pipeline can be tested by running `pytest`. We use [mypy](https://github.com/python/mypy) for type checking.
This can be run by running `mypy src` (this runs mypy on the `src` directory).

## Acknowledgements <a name="acknowledgements"></a>
Huge thanks to @ECMWF for making this project possible!
