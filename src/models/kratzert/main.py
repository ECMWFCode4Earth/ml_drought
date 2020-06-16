import numpy as np
import torch

from src.preprocess.camels_kratzert import (
    CAMELSCSV,
    get_basins,
    RunoffEngineer,
    CamelsH5,
)
from . import Model


def _prepare_data(
    data_dir: Path,
    basins: List[int],
    train_dates: List[int],
    with_basin_str: bool = True,
    target_var: str = "discharge_spec",
    x_variables: Optional[List[str]] = ["precipitation", "peti"],
    static_variables: Optional[List[str]] = None,
    ignore_static_vars: Optional[List[str]] = None,
    seq_length: int = 365,
    with_static: bool = True,
    concat_static: bool = False,
):
    engineer = RunoffEngineer(
        data_dir=data_dir,
        basins=basins,
        train_dates=train_dates,
        with_basin_str=with_basin_str,
        target_var=target_var,
        x_variables=x_variables,
        static_variables=static_variables,
        ignore_static_vars=None,
        seq_length=seq_length,
        with_static=with_static,
        concat_static=concat_static,
    )

    engineer.create_training_data()


def train(
    data_dir: Path,
    basins: List[int],
    train_dates: List[int],
    with_basin_str: bool = True,
    target_var: str = "discharge_spec",
    x_variables: Optional[List[str]] = ["precipitation", "peti"],
    static_variables: Optional[List[str]] = None,
    ignore_static_vars: Optional[List[str]] = None,
    seq_length: int = 365,
    with_static: bool = True,
    concat_static: bool = False,
    seed: int = 10101,
    cache: bool = True,
    batch_size: int = 32,
    num_workers: int = 1,
):
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    basins = get_basins(data_dir )

    # engineer the data for this training run
    _prepare_data(
        data_dir=data_dir,
        basins=basins,
        train_dates=train_dates,
        with_basin_str=with_basin_str,
        target_var=target_var,
        x_variables=x_variables,
        static_variables=static_variables,
        ignore_static_vars=ignore_static_vars,
        seq_length=seq_length,
        with_static=with_static,
        concat_static=concat_static,
    )

    # create dataloader
    data = CamelsH5(
        data_dir=data_dir,
        basins=basins,
        concat_static=concat_static,
        cache=cache,
        with_static=with_static,
        train_dates=train_dates,
    )

    input_size_stat
    input_size_dyn

    loader = DataLoader(
        data, batch_size=batch_size,
        shuffle=True, num_workers=num_workers
    )