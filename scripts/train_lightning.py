import sys

sys.path.append("..")

from src.models import (
    load_model,
)
from src.lightning_models import LightningModel

from scripts.utils import get_data_path
from argparse import Namespace
import pytorch_lightning as pl


if __name__ == "__main__":
    always_ignore_vars = [
        "VCI",
        "p84.162",
        "sp",
        "tp",
        "Eb",
        "VCI1M",
        "RFE1M",
        "VCI3M",
        # "boku_VCI",
        "modis_ndvi",
        "SMroot",
        "lc_class",  # remove for good clustering (?)
        "lc_class_group",  # remove for good clustering (?)
        "slt", # remove for good clustering (?)
        "no_data_one_hot",
        "lichens_and_mosses_one_hot",
        "permanent_snow_and_ice_one_hot",
        "urban_areas_one_hot",
        "water_bodies_one_hot",
        "t2m",
        "SMsurf",
        # "pev",
        # "e",
        "E",
    ]
    target_vars = ["boku_VCI"]
    dynamic_vars = ["precip", "t2m", "pet", "E", "SMsurf"]
    static = True

    hparams = Namespace(
        **{
            "model_name": "EALSTM",
            "data_path": get_data_path(),
            "experiment": "one_month_forecast",
            "hidden_size": 64,
            "rnn_dropout": 0.3,
            "include_latlons": True,
            "static_embedding_size": 64,
            "include_prev_y": False,
            "include_yearly_aggs": False,
            "static": "features",
            "batch_size": 1,
            "include_pred_month": True,
            "pred_months": None,
            "ignore_vars": always_ignore_vars,
            "include_monthly_aggs": False,
            "surrounding_pixels": None,
            "predict_delta": False,
            "spatial_mask": None,
            "normalize_y": True,
            "dense_features": [128],
            "val_ratio": 0.3,
            "learning_rate": 1e3,
            "save_preds": True,
            "static": False,
        }
    )

    model = LightningModel(hparams)
    kwargs = dict(fast_dev_run=True)  # , gpus=[0]
    model.fit(**kwargs)

    # TODO: add list of static vars that are included to the ModelArrays
    # TODO: get the model running on real data (with gpu)
