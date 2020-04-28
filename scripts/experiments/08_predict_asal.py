from _base_models import persistence, regression, linear_nn, rnn, earnn
from scripts.utils import _rename_directory, get_data_path
import sys

sys.path.append("../..")


if __name__ == "__main__":
    important_vars = ["VCI", "precip", "t2m", "pev", "E", "SMsurf", "SMroot"]
    always_ignore_vars = ["ndvi", "p84.162", "sp", "tp", "Eb"]

    # persistence()
    # regression(ignore_vars=always_ignore_vars, predict_delta=True)
    # gbdt(ignore_vars=always_ignore_vars, predict_delta=True)
    # linear_nn(ignore_vars=always_ignore_vars, predict_delta=True)
    # rnn(ignore_vars=always_ignore_vars, predict_delta=True)
    earnn(
        experiment="one_month_forecast",
        include_pred_month=True,
        surrounding_pixels=None,
        pretrained=False,
        explain=False,
        static="features",
        ignore_vars=always_ignore_vars,
        num_epochs=50,  # 50
        early_stopping=5,  # 5
        hidden_size=256,
        static_embedding_size=64,
        predict_delta=True,
        spatial_mask="data/interim/boundaries_preprocessed/kenya_asal_mask.nc",
    )

    # rename the output file
    data_path = get_data_path()

    _rename_directory(
        from_path=data_path / "models" / "one_month_forecast",
        to_path=data_path / "models" / "one_month_forecast_predict_delta_ASAL",
        with_datetime=False,
    )
