import pandas as pd

from .base import ModelBase

from ray import tune
from ray.tune import track
from ray.tune.schedulers import ASHAScheduler

from typing import Any, Dict


def tune_model(model: ModelBase, search_space: Dict[str, Any]) -> pd.DataFrame:

    def _train(config: Dict[str, Any]) -> None:
        train_loss, val_loss = model.train(**config)  # type: ignore
        track.log(train_loss=train_loss, val_loss=val_loss)

    analysis = tune.run(_train, num_samples=30,
                        scheduler=ASHAScheduler(metric="val_loss", mode="min"),
                        config=search_space)
    return analysis.trial_dataframes
