import xarray as xr
from pathlib import Path
import numpy as np
import sys
from typing import (Tuple, List, Union, )

sys.path.append("..")

from src.analysis import AdministrativeRegionAnalysis
from src.analysis import LandcoverRegionAnalysis
from scripts.utils import get_data_path
from src.analysis import read_train_data, read_test_data, read_pred_data
from src.utils import get_ds_mask
from src.models import load_model
from src.models.neural_networks.base import NNBase


def run_administrative_region_analysis():
    # if the working directory is alread ml_drought don't need ../data
    if Path(".").absolute().as_posix().split("/")[-1] == "ml_drought":
        data_path = Path("data")
    else:
        data_path = Path("../data")

    assert [
        f for f in (data_path / "features").glob("*/test/*/*.nc")
    ] != [], "There is no true data (has the pipeline been run?)"
    assert [
        f for f in (data_path / "models").glob("*/*/*.nc")
    ] != [], "There is no model data (has the pipeline been run?)"
    assert [f for f in (data_path / "analysis").glob("*/*.nc")] != [], (
        "There are no processed regions. " "Has the pipeline been run?"
    )

    analyzer = AdministrativeRegionAnalysis(data_path)
    analyzer.analyze()
    print(analyzer.regional_mean_metrics)


def run_landcover_region_analysis():
    # if the working directory is alread ml_drought don't need ../data
    if Path(".").absolute().as_posix().split("/")[-1] == "ml_drought":
        data_path = Path("data")
    else:
        data_path = Path("../data")

    assert [
        f for f in (data_path / "features").glob("*/test/*/*.nc")
    ] != [], "There is no true data (has the pipeline been run?)"
    assert [
        f for f in (data_path / "models").glob("*/*/*.nc")
    ] != [], "There is no model data (has the pipeline been run?)"
    assert [
        f.name
        for f in (
            data_path / "interim" / "static" / "esa_cci_landcover_preprocessed"
        ).glob("*.nc")
    ] != [], ("There is no landcover data. " "Has the pipeline been run?")

    analyzer = LandcoverRegionAnalysis(data_path)
    analyzer.analyze()
    print(analyzer.regional_mean_metrics)


def read_all_data(data_dir: Path, experiment="one_month_forecast", static: bool = False) -> Tuple[xr.Dataset]:
    X_train, y_train = read_train_data(data_dir, experiment=experiment)
    X_test, y_test = read_test_data(data_dir, experiment=experiment)

    if static:
        static_ds = xr.open_dataset(data_dir / "features/static/data.nc")
    return (X_train, y_train, X_test, y_test)


def read_all_available_pred_data(data_dir: Path, experiment: str = "one_month_forecast") -> List[xr.Dataset]:
    model_dir = (data_dir / f"models/{experiment}")
    models = np.array([d.name for d in model_dir.iterdir()])

    # drop the models that haven't written .nc files
    models_run_bool = [
        any([".nc" in file.name for file in (model_dir / model).iterdir()])
        for model in models
    ]
    models = models[models_run_bool]

    return {model: read_pred_data(model, data_dir, experiment=experiment)[-1] for model in models}


def load_nn(data_dir: Path, model_str: str, experiment: str = "one_month_forecast") -> NNBase:
    return load_model(data_dir / f"models/{experiment}/{model_str}/model.pt")


def plot_predictions():

    return


if __name__ == "__main__":
    # run_administrative_region_analysis()
    # run_landcover_region_analysis()
    data_dir = get_data_path()
    experiment = "one_month_forecast"

    # read train/test data
    X_train, y_train, X_test, y_test = read_all_data(
        data_dir,
        static=False,
        experiment=experiment,
    )
    mask = get_ds_mask(X_train.VCI)

    # read the predicted data
    predictions = read_all_available_pred_data(
        data_dir,
        experiment=experiment,
    )

    # read in the models (pytorch models)
    models_str = [m for m in predictions.keys()]
    models = {}
    for m in [m for m in models_str if m != "previous_month"]:
        models[m] = load_nn(data_dir, m, experiment=experiment)


preds = predictions['rnn']
model = 'rnn'
read_pred_data(model, data_dir, experiment=experiment)