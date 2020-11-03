from collections import defaultdict
from typing import Tuple
from neuralhydrology.utils.errors import NoTrainDataError
from neuralhydrology.evaluation import RegressionTester
from neuralhydrology.modelzoo.basemodel import BaseModel
from tqdm import tqdm
import numpy as np
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from typing import DefaultDict

# load in config
run_dir = data_dir / "runs/ensemble_EALSTM/ealstm_ensemble6_nse_1998_2008_2910_030601"

# Config file
config_path = (run_dir / "config.yml")
config = Config(config_path)



def get_states_from_forward(model: BaseModel, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    all_hidden_states = []
    all_cell_states = []
    # For all the basin data in Loader
    for basin_data in loader:
        with torch.no_grad():
            predict = model.forward(basin_data)
            all_hidden_states.append(predict["h_n"].detach().numpy())
            all_cell_states.append(predict["c_n"].detach().numpy())

    basin_hidden_states = np.vstack(all_hidden_states)
    basin_cell_states = np.vstack(all_cell_states)

    return basin_hidden_states, basin_cell_states

## CONVERT to
Tester = RegressionTester(cfg=config, run_dir=run_dir, period="test", init_model=True)
all_basins = load_basin_file(Path(config.train_basin_file))

all_basin_data = defaultdict(dict)

# For each basin create a DataLoader
for ix, basin in enumerate(tqdm(all_basins)):
    try:
        ds = Tester._get_dataset(basin)
    except NoTrainDataError:
        print(f"{basin} Missing")
        continue

    loader = DataLoader(ds, batch_size=config.batch_size, num_workers=0)
    basin_hidden_states, basin_cell_states = get_states_from_forward(model, loader)
    all_basin_data[basin]["h_s"] = basin_hidden_states
    all_basin_data[basin]["c_s"] = basin_cell_states
    if ix == 2:
        break


print("****** DONE DONE DONE *******")


# input_data = cs_data
input_data = cs_data.groupby("time").mean()
target_data = dynamic["discharge_spec"]

from typing import Any


class CellStateDataset(Dataset):
    def __init__(self, input_data: xr.Dataset, target_data: xr.DataArray, config, mean: bool = True):
        assert all(np.isin(["time", "dimension", "station_id"], input_data.dims))
        assert "cell_state" in input_data
        assert all(np.isin(input_data.station_id.values, target_data.station_id.values))

        self.input_data = input_data

        # test times
        test_times = pd.date_range(config.test_start_date, config.test_end_date, freq="D")
        bool_times = np.isin(self.input_data.time.values, test_times)

        # get input/target data
        self.input_data = self.input_data.sel(time=bool_times)
        self.test_times = self.input_data.time.values
        self.target_data = target_data.sel(time=self.test_times)

        # basins
        self.basins = input_data.station_id.values

        # dimensions
        self.dimensions = len(input_data.dimension.values)

        # create x y pairs
        self.create_samples()

    def __len__(self):
        return len(self.samples)

    def create_samples(self):
        self.samples = []
        self.basin_samples = []
        self.time_samples = []

        for basin in self.basins:
            # read the basin data
            X = self.input_data["cell_state"].sel(station_id=basin).values
            Y = self.target_data.sel(station_id=basin).values

            finite_indices = np.logical_and(np.isfinite(Y), np.isfinite(X).all(axis=1))
            X, Y = X[finite_indices], Y[finite_indices]
            times = self.input_data["time"].values[finite_indices].astype(float)

            # convert to Tensors
            X = torch.from_numpy(X).float().to("cuda:0")
            Y = torch.from_numpy(Y).float().to("cuda:0")


            # create unique samples [(64,), (1,)]
            samples = [(x, y.reshape(-1)) for (x, y) in zip(X, Y)]
            self.samples.extend(samples)
            self.basin_samples.extend([basin for _ in range(len(samples))])
            self.time_samples.extend(times)


    def __getitem__(self, item: int) -> Tuple[Tuple[str, Any], Tuple[torch.Tensor]]:
        basin = str(self.basin_samples[item])
        time = self.time_samples[item]
        x, y = self.samples[item]

        return (basin, time), (x, y)


dataset = CellStateDataset(input_data=input_data, target_data=dynamic["discharge_spec"], config=config)
loader = DataLoader(dataset, batch_size=256, shuffle=True)
"""

#  One linear layer model to map
D_in = 64
model = torch.nn.Sequential(torch.nn.Linear(D_in, 1))

#  loss function
loss_fn = torch.nn.MSELoss(reduction="sum")

# optimizer
learning_rate = 1e-2
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#  TRAIN
losses = []
for t in range(500):
    y_pred = model(X)
    loss = loss_fn(y_pred, y)

    # train/update the weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.detach().cpu().numpy())
"""
