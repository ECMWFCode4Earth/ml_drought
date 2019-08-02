# Scripts

These scripts are the practical entrypoints into the pipeline. 
In the future, they will be replaced by the [`run.py`](../run.py) file, with configurations defined by `.json` objects.

The scripts match the major steps of the pipeline, which are also described in more detail in the 
[notebooks documentation](../notebooks/docs). 
In nearly all cases, the classes and public functions are documented using docstrings. This documentation can be exposed
using [`help`](https://docs.python.org/3/library/functions.html#help).

Running all the current scripts (which would roughly equate to running the pipeline end to end) requires roughly
**500 GB** of disk space. We can successfully run all the steps using a google cloud instance with **13GB** of memory.

### 1. [export.py](export.py)

This script is responsible for exporting data from the various data-stores. A notebook with more details can be found
[here](../notebooks/docs/01_Exporters.ipynb).

All the arguments being passed to the exporters in the scripts are what is currently being used for our experiments.

### 2. [preprocess.py](preprocess.py)

This script is responsible for taking all the exported data and ensuring it has a uniform format (i.e. dimensions have
the same names, all the data is on the same grid).

We currently regrid onto the CHIRPS dataset, since this was the dataset with the lowest resolution, but plan on regridding
instead onto VHI (the target variable). This is defined by passing a `regrid_path` argument to the preprocessors.

### 3. [engineer.py](engineer.py)

This script is responsible for preparing the data so that it can be input into the machine learning models. Specifically,
it does the following:

* Breaks the data down into `x.nc`, which the model sees, and `y.n`, which the model tries to predict
* Splits the data into a `train` and `test` set
* Processes the static data

Two experiments can be run: `one_month_forecast`, where we use data up to month `n-1` to predict a target variable
(vegetation health by default) at month `n`, or `nowcast`, where we use data up to month `n` to predict a target variable
in month `n`.

### 4. [models.py](models.py)

This script trains and saves the models.
