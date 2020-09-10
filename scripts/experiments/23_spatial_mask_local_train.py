import pandas as pd
import xarray as xr
import sys

sys.path.append("../..")

from src.models import RecurrentNetwork
from scripts.analysis import extract_json_results_dict
from scripts.utils import _rename_directory
from scripts.utils import get_data_path


def lstm(
    experiment="one_month_forecast",
    include_pred_month=True,
    surrounding_pixels=None,
    ignore_vars=None,
    pretrained=False,
    static=None,
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
    )
    return predictor


if __name__ == "__main__":
    # create region experiment data directory
    data_dir = get_data_path()
    expt_dir = data_dir / "models/region_expt"
    if not expt_dir.exists():
        expt_dir.mkdir(exist_ok=True, parents=True)

    # load in the region netcdf file
    region_nc = xr.open_dataset(
        data_dir / "analysis/boundaries_preprocessed/state_l1_india.nc"
    )

    keys = [int(i) for i in region_nc.attrs["keys"].split(",")]
    state_names = [r.rstrip().lstrip() for r in region_nc.attrs["values"].split(",")]

    # init list to store 'total_rmse' for each state
    total_rmse = []

    # --- TRAIN AND EVALUATE THE MODEL FOR EACH STATE (48 times) --- #
    for state_name, state_key in zip(state_names, keys):
        print(f"** Starting Analysis on {state_name} **")
        save_name = "_".join(state_name.lower().split(" "))
        mask = region_nc.where(region_nc["state_l1"] == state_key)["state_l1"]

        # initialise model
        model = lstm()

        # train the model on that spatial subset
        model.train(num_epochs=1, early_stopping=5, spatial_mask=mask, verbose=True)
        model.evaluate(save_preds=True)
        model.save_model()

        total, df = extract_json_results_dict(
            model="rnn", experiment="one_month_forecast"
        )
        df["region"] = [state_name for _ in range(len(df))]
        total_rmse.append(total)

        df.to_csv(expt_dir / f"{save_name}_rnn_area_specific.csv")

        to_path = expt_dir / f"{save_name}_{model.model_dir.name}"
        _rename_directory(
            from_path=model.model_dir, to_path=to_path,
        )

        print(
            f"** LSTM Trained and Evaluated for state: {state_name}. RMSE: {total_rmse} **"
        )
        print()

        break

    total_df = pd.DataFrame(total_rmse, index=state_names)
    total_df.to_csv(expt_dir / "region_results.csv")
