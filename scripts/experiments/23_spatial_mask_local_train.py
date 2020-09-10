import pandas as pd
import xarray as xr
import sys

sys.path.append("../..")

from src.models import RecurrentNetwork
from scripts.analysis import extract_json_results_dict
from scripts.utils import _rename_directory, get_data_path
from scripts.models import get_forecast_vars, get_ignore_static_vars


def lstm(
    experiment="one_month_forecast",
    include_pred_month=True,
    surrounding_pixels=None,
    ignore_vars=None,
    pretrained=False,
    static="features",
    spatial_mask=None
):
    data_path = get_data_path()
    predictor = RecurrentNetwork(
        hidden_size=128,
        data_folder=data_path,
        experiment=experiment,
        include_pred_month=include_pred_month,
        surrounding_pixels=surrounding_pixels,
        ignore_vars=ignore_vars,
        static=static,
        spatial_mask=spatial_mask,
    )
    return predictor


if __name__ == "__main__":
    # create region experiment data directory
    data_dir = get_data_path()
    expt_dir = data_dir / "models/region_expt"
    if not expt_dir.exists():
        expt_dir.mkdir(exist_ok=True, parents=True)

    # ignore vars
    forecast_vars = get_forecast_vars()
    ignore_static_vars = get_ignore_static_vars()
    ignore_vars = forecast_vars + ignore_static_vars

    # load in the region netcdf file
    region_nc = xr.open_dataset(
        data_dir / "analysis/boundaries_preprocessed/state_l1_india.nc"
    )

    keys = [int(i) for i in region_nc.attrs["keys"].split(",")]
    state_names = [r.rstrip().lstrip() for r in region_nc.attrs["values"].split(",")]

    # init list to store 'total_rmse' for each state
    total_rmse = []
    run_names = []

    # --- TRAIN AND EVALUATE THE MODEL FOR EACH STATE (48 times) --- #
    for state_name, state_key in zip(state_names, keys):
        print(f"** Starting Training on {state_name} **")
        save_name = "_".join(state_name.lower().split(" "))
        nanmask = region_nc.where(region_nc["state_l1"] == state_key)["state_l1"]
        mask = nanmask.isnull()

        # initialise model
        model = lstm(spatial_mask=mask)

        # train the model on that spatial subset
        model.train(num_epochs=1, early_stopping=5, verbose=True)
        model.evaluate(save_preds=True)
        model.save_model()

        total, df = extract_json_results_dict(
            data_dir=data_dir,
            model="rnn",
            experiment="one_month_forecast"
        )
        df["region"] = [state_name for _ in range(len(df))]

        # append values to lists
        total_rmse.append(total)
        run_names.append(state_name)

        df.to_csv(expt_dir / f"{save_name}_rnn_area_specific.csv")

        to_path = expt_dir / f"{save_name}_{model.model_dir.name}"
        _rename_directory(
            from_path=model.model_dir,
            to_path=to_path,
            with_datetime=True
        )

        print(
            f"** LSTM Trained and Evaluated for state: {state_name}. RMSE: {total_rmse} **"
        )
        print()

    total_df = pd.DataFrame(total_rmse, index=run_names).rename({0: "rmse"}, axis=1)
    total_df.to_csv(expt_dir / "region_results.csv")
