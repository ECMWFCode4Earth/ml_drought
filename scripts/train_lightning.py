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
    all_vars = [
        "VCI",
        "precip",
        "t2m",
        "pev",
        "E",
        "SMsurf",
        "SMroot",
        "Eb",
        "sp",
        "tp",
        "ndvi",
        "p84.162",
        "boku_VCI",
        "VCI3M",
        "modis_ndvi",
    ]
    target_vars = ["boku_VCI"]
    dynamic_vars = ["precip", "t2m", "pet", "E", "SMsurf"]
    ignore_vars = [v for v in all_vars if v not in dynamic_vars]
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
            "ignore_vars": ignore_vars,
            "include_monthly_aggs": False,
            "surrounding_pixels": None,
            "predict_delta": False,
            "spatial_mask": None,
            "normalize_y": True,
            "dense_features": [128],
            "val_ratio": 0.3,
            "learning_rate": 1e3,
            "save_preds": True,
        }
    )

    model = LightningModel(hparams)
    kwargs = dict(fast_dev_run=True)
    model.fit(**kwargs)
