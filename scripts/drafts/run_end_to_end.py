from pathlib import Path
import sys

# get the project directory
if Path(".").absolute().name == "ml_drought":
    sys.path.append(".")
elif Path(".").absolute().parents[0].name == "ml_drought":
    sys.path.append("..")
elif Path(".").absolute().parents[1].name == "ml_drought":
    sys.path.append("../..")
else:
    assert False, "We cant find the base project directory `ml_drought`"

# 1. exporters
from src.exporters import (
    ERA5Exporter,
    VHIExporter,
    CHIRPSExporter,
    ERA5ExporterPOS,
    GLEAMExporter,
    ESACCIExporter,
)
from src.exporters.srtm import SRTMExporter

# 2. preprocessors
from src.preprocess import (
    VHIPreprocessor,
    CHIRPSPreprocessor,
    PlanetOSPreprocessor,
    GLEAMPreprocessor,
    ESACCIPreprocessor,
)
from src.preprocess.srtm import SRTMPreprocessor

# 3. engineer
from src.engineer import Engineer

# 4. model
from src.models import (
    Persistence,
    LinearRegression,
    LinearNetwork,
    RecurrentNetwork,
    EARecurrentNetwork,
)
from src.models.data import DataLoader

# 5. analysis
from src.analysis import plot_shap_values

# ----------------------------------------------------------------
# Export (download the data)
# ----------------------------------------------------------------


def export_data(data_path):
    # target variable
    print("** Exporting VHI **")
    exporter = VHIExporter(data_path)
    exporter.export()
    del exporter

    # precip
    print("** Exporting CHIRPS Precip **")
    exporter = CHIRPSExporter(data_path)
    exporter.export(years=None, region="global", period="monthly")
    del exporter

    # temperature
    print("** Exporting ERA5 Temperature **")
    exporter = ERA5Exporter(data_path)
    exporter.export(variable="2m_temperature", granularity="monthly")
    del exporter

    # evaporation
    print("** Exporting GLEAM Evaporation **")
    exporter = GLEAMExporter(data_folder=data_path)
    exporter.export(["E"], "monthly")
    del exporter

    # topography
    print("** Exporting SRTM Topography **")
    exporter = SRTMExporter(data_folder=data_path)
    exporter.export()
    del exporter

    # landcover
    print("** Exporting Landcover **")
    exporter = ESACCIExporter(data_folder=data_path)
    exporter.export()
    del exporter


# ----------------------------------------------------------------
# Preprocess
# ----------------------------------------------------------------


def preprocess_data(data_path):
    # preprocess VHI
    print("** Preprocessing VHI **")
    processor = VHIPreprocessor(data_path)
    processor.preprocess(
        subset_str="kenya",
        regrid=regrid_path,
        n_parallel_processes=1,
        resample_time="M",
        upsampling=False,
    )

    regrid_path = data_path / "interim" / "vhi_preprocessed" / "vhi_kenya.nc"

    # preprocess CHIRPS Rainfall
    print("** Preprocessing CHIRPS Precipitation **")
    processor = CHIRPSPreprocessor(data_path)
    processor.preprocess(subset_str="kenya", regrid=regrid_path, n_parallel_processes=1)

    # preprocess GLEAM evaporation
    print("** Preprocessing GLEAM Evaporation **")
    processor = GLEAMPreprocessor(data_path)
    processor.preprocess(
        subset_str="kenya", regrid=regrid_path, resample_time="M", upsampling=False
    )

    # preprocess SRTM Topography
    print("** Preprocessing SRTM Topography **")
    processor = SRTMPreprocessor(data_path)
    processor.preprocess(subset_str="kenya", regrid=regrid_path)

    # preprocess ESA CCI Landcover
    print("** Preprocessing ESA CCI Landcover **")
    processor = ESACCIPreprocessor(data_path)
    processor.preprocess(
        subset_str="kenya", regrid=regrid_path, resample_time="M", upsampling=False
    )


# ----------------------------------------------------------------
# Engineer (train/test split for each experiment)
# ----------------------------------------------------------------


def engineer(
    data_path,
    experiment="one_month_forecast",
    process_static=True,
    pred_months=12,
    expected_length=12,
):
    engineer = Engineer(data_path, experiment=experiment, process_static=process_static)
    engineer.engineer(
        test_year=2018,
        target_variable="VHI",
        pred_months=pred_months,
        expected_length=pred_months,
    )


def engineer_data(data_path):
    print("** Engineering data for one_month_forecast experiment **")
    engineer(data_path=data_path, experiment="one_month_forecast")

    print("** Engineering data for nowcast experiment **")
    engineer(data_path=data_path, experiment="nowcast")


# ----------------------------------------------------------------
# Run the actual models
# ----------------------------------------------------------------


def run_models(data_path, experiment):
    # NOTE: this model is the same for all experiments
    print("Running persistence model")
    predictor = Persistence(data_path, experiment=experiment)
    predictor.evaluate(save_preds=True)

    # linear regression
    print(f"Running Linear Regression model: {experiment}")
    predictor = LinearRegression(
        data_path, experiment=experiment, include_pred_month=True, surrounding_pixels=1
    )
    predictor.train(num_epochs=10, early_stopping=3)

    # linear network
    print(f"Running Linear Neural Network model: {experiment}")
    predictor = LinearNetwork(
        data_folder=data_path,
        experiment=experiment,
        layer_sizes=[100],
        include_pred_month=True,
        surrounding_pixels=1,
    )
    predictor.train(num_epochs=10, early_stopping=3)
    predictor.evaluate(save_preds=True)
    predictor.save_model()

    # recurrent network
    print(f"Running RNN (LSTM) model: {experiment}")
    predictor = RecurrentNetwork(
        data_folder=data_path,
        hidden_size=128,
        experiment=experiment,
        include_pred_month=True,
        surrounding_pixels=1,
    )
    predictor.train(num_epochs=10, early_stopping=3)
    predictor.evaluate(save_preds=True)
    predictor.save_model()

    # EA LSTM
    print(f"Running Entity Aware LSTM model: {experiment}")
    predictor = EARecurrentNetwork(
        data_folder=data_path,
        hidden_size=128,
        experiment=experiment,
        include_pred_month=True,
        surrounding_pixels=1,
    )
    predictor.train(num_epochs=10, early_stopping=3)
    predictor.evaluate(save_preds=True)
    predictor.save_model()


def model_data(data_path):
    experiments = ["one_month_forecast", "nowcast"]
    for experiment in experiments:
        print("** Running Experiments **")
        run_models(data_path, experiment)


if __name__ == "__main__":
    # get the data path
    if Path(".").absolute().name == "ml_drought":
        data_path = Path("data")
    elif Path(".").absolute().parents[0].name == "ml_drought":
        data_path = Path("../data")
    elif Path(".").absolute().parents[1].name == "ml_drought":
        data_path = Path("../../data")
    else:
        assert False, "We cant find the base package directory `ml_drought`"

    # export
    print("** Running Exporters **")
    export_data(data_path)

    # preprocess
    print("** Running Preprocessors **")
    preprocess_data(data_path)

    # engineer
    print("** Running Engineer **")
    engineer_data(data_path)

    # models
    print("** Running Models **")
    model_data(data_path)
