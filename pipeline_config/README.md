# Pipeline configurations

The pipeline is configured using `json` objects - see the [minimal configuration](minimal.json) for
an example. The pipeline consists of the following steps:

#### 1. Export

The export step consists of downloading data from remote datasets to a local folder.

The format for exporter configurations is `{dataset: [list of variable configurations]}`.

#### 2. Preprocess

Preprocess exported files.

#### 3. Engineer

Engineer the preprocessed files into a `.nc` files ready to be input into the machine learning
models.

#### 4. Models

Train models

## Implemented configurations

The following configurations have been implemented:

#### [minimal](minimal.json)

A minimal example configuration, in which VHI and ERA5 data are used. The VHI data is put through
the persistence persistence model.

#### [era5_forecast](era5_forecast.json)

A variety of datasets are downloaded and regridded onto the ERA5 grid. Those datasets are then
used to train an EA-LSTM to predict vegetation health one month in the future.
