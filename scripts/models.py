import sys

sys.path.append("..")

from src.models import (
    Persistence,
    LinearRegression,
    LinearNetwork,
    RecurrentNetwork,
    EARecurrentNetwork,
    load_model,
)
from src.analysis import all_explanations_for_file

from scripts.utils import get_data_path


def ignore_vars_previous():
    ignore_vars = ['VCI', 'p84.162', 'sp', 'tp', 'Eb', 'VCI1M', 'RFE1M', 'boku_VCI', 'modis_ndvi', 'SMroot', 'lc_class', 'lc_class_group', 'no_data_one_hot', 'lichens_and_mosses_one_hot', 'permanent_snow_and_ice_one_hot', 'urban_areas_one_hot', 'water_bodies_one_hot', 't2m', 'SMsurf', 'E']


def persistence(
    experiment="one_month_forecast",
    data_path = None,
):
    if data_path is None:
        data_path = get_data_path()
    spatial_mask = data_path / "interim/boundaries_preprocessed/kenya_asal_mask.nc"
    spatial_mask = None
    predictor = Persistence(data_path, experiment=experiment, spatial_mask=spatial_mask)
    predictor.evaluate(save_preds=True)

    return predictor


def regression(
    experiment="one_month_forecast",
    include_pred_month=True,
    surrounding_pixels=None,
    ignore_vars=None,
    data_path = None,
):
    if data_path is None:
        data_path = get_data_path()
    spatial_mask = data_path / "interim/boundaries_preprocessed/kenya_asal_mask.nc"
    spatial_mask = None

    predictor = LinearRegression(
        data_path,
        experiment=experiment,
        include_pred_month=include_pred_month,
        surrounding_pixels=surrounding_pixels,
        ignore_vars=ignore_vars,
        static="embeddings",
        spatial_mask=spatial_mask,
    )
    predictor.train()
    predictor.evaluate(save_preds=True)

    # mostly to test it works
    # predictor.explain(save_shap_values=True)

    return predictor


def linear_nn(
    experiment="one_month_forecast",
    include_pred_month=True,
    surrounding_pixels=None,
    ignore_vars=None,
    pretrained=False,
    static=None,
    data_path = None,
    include_latlons=False, 
    hidden_size: int = 128,
):
    if data_path is None:
        data_path = get_data_path()

    predictor = LinearNetwork(
        layer_sizes=[hidden_size],
        data_folder=data_path,
        experiment=experiment,
        include_pred_month=include_pred_month,
        surrounding_pixels=surrounding_pixels,
        ignore_vars=ignore_vars,
        static=static,
        include_latlons=include_latlons,
    )
    predictor.train(num_epochs=50, early_stopping=5)
    predictor.evaluate(save_preds=True)
    predictor.save_model()

    # _ = predictor.explain(save_shap_values=True)

    return predictor


def rnn(
    experiment="one_month_forecast",
    include_pred_month=True,
    include_latlons=False, 
    surrounding_pixels=None,
    ignore_vars=None,
    pretrained=True,
    static=None,
    data_path = None,
    calculate_shap: bool = False,
    hidden_size: int = 128,
):
    if data_path is None:
        data_path = get_data_path()
    if not pretrained:
        predictor = RecurrentNetwork(
            hidden_size=hidden_size,
            data_folder=data_path,
            experiment=experiment,
            include_pred_month=include_pred_month,
            surrounding_pixels=surrounding_pixels,
            ignore_vars=ignore_vars,
            static=static,
            include_latlons=include_latlons,
        )
        predictor.train(num_epochs=50, early_stopping=5)
        predictor.evaluate(save_preds=True)
        predictor.save_model()
    else:
        predictor = load_model(data_path / f"models/{experiment}/rnn/model.pt")

    # if calculate_shap:
    #     _ = predictor.explain(save_shap_values=True)

    test_file = data_path / f"features/{experiment}/test/2018_3"
    assert test_file.exists()
    if calculate_shap:
        all_explanations_for_file(test_file, predictor, batch_size=100)

    return predictor


def earnn(
    experiment="one_month_forecast",
    include_pred_month=True,
    include_latlons=False, 
    surrounding_pixels=None,
    pretrained=True,
    ignore_vars=None,
    static="embeddings",
    data_path = None,
    calculate_shap: bool = False,
    hidden_size: int = 128,
):
    if data_path is None:
        data_path = get_data_path()

    if static is None:
        print("** Cannot fit EALSTM without spatial information **")
        return

    if not pretrained:
        predictor = EARecurrentNetwork(
            hidden_size=hidden_size,
            data_folder=data_path,
            experiment=experiment,
            include_pred_month=include_pred_month,
            surrounding_pixels=surrounding_pixels,
            ignore_vars=ignore_vars,
            static=static,
            include_latlons=include_latlons,
        )
        print(f"Nfeatures per month: {predictor.features_per_month}")
        predictor.train(num_epochs=50, early_stopping=5)
        predictor.evaluate(save_preds=True)
        predictor.save_model()
    else:
        predictor = load_model(data_path / f"models/{experiment}/ealstm/model.pt")

    test_file = data_path / f"features/{experiment}/test/2018_3"
    assert test_file.exists()
    
    if calculate_shap:
        all_explanations_for_file(test_file, predictor, batch_size=100)

    return predictor


if __name__ == "__main__":
    ignore_vars = ["VCI", "p84.162", "sp", "tp", "VCI1M", "modis_ndvi", "E", "Eb", "SMroot", "SMsurf"]
    data_dir = get_data_path()
    drought_data_dir = data_dir / "VEG_DATA"
    other_data_dir = data_dir / "DROUGHT"

    # persistence()
    # regression(ignore_vars=ignore_vars)
    # linear_nn(ignore_vars=ignore_vars, static='features')
    rnn(ignore_vars=ignore_vars, static='features', data_path=other_data_dir, pretrained=False)
    # earnn(ignore_vars=ignore_vars, static='features')
