from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from random import shuffle as shuffle_list

from ..data import DataLoader, _BaseIter, ModelArrays, TrainData

from typing import Union, Tuple, Optional, List, Iterable


class TripletLoader(DataLoader):
    def __init__(
        self,
        neighbouring_distance: Union[float, Tuple[float, float]] = 2,
        multiplier: int = 10,
        data_path: Path = Path("data"),
        batch_file_size: int = 1,
        shuffle_data: bool = True,
        clear_nans: bool = True,
        normalize: bool = True,
        experiment: str = "one_month_forecast",
        mask: Optional[List[bool]] = None,
        pred_months: Optional[List[int]] = None,
        to_tensor: bool = False,
        surrounding_pixels: Optional[int] = None,
        ignore_vars: Optional[List[str]] = None,
        monthly_aggs: bool = True,
        static: Optional[str] = "features",
        device: str = "cpu",
    ) -> None:
        super().__init__(
            data_path,
            batch_file_size,
            "train",  # we only want to load the training data
            shuffle_data,
            clear_nans,
            normalize,
            experiment,
            mask,
            pred_months,
            to_tensor,
            surrounding_pixels,
            ignore_vars,
            monthly_aggs,
            static,
            device,
        )
        if isinstance(neighbouring_distance, float) or isinstance(
            neighbouring_distance, int
        ):
            neighbouring_distance = (neighbouring_distance, neighbouring_distance)
        self.neighbouring_distance = neighbouring_distance
        self.multiplier = multiplier

    def __iter__(self):
        return _TripletIter(self)

    def __len__(self) -> int:
        return len(self.data_files) // self.batch_file_size


class _TripletIter(_BaseIter):
    def __init__(self, loader: TripletLoader) -> None:
        super().__init__(loader)
        self.neighbouring_distance = loader.neighbouring_distance
        self.multiplier = loader.multiplier

    def __next__(
        self,
    ) -> Tuple[
        Tuple[Optional[torch.Tensor], ...],
        Tuple[Optional[torch.Tensor], ...],
        Tuple[Optional[torch.Tensor], ...],
    ]:

        global_modelarrays: Optional[ModelArrays] = None

        if self.idx < self.max_idx:

            cur_max_idx = min(self.idx + self.batch_file_size, self.max_idx)
            while self.idx < cur_max_idx:
                subfolder = self.data_files[self.idx]
                arrays = self.ds_folder_to_np(
                    subfolder, clear_nans=self.clear_nans, to_tensor=False
                )
                if arrays.x.historical.shape[0] == 0:
                    print(f"{subfolder} returns no values. Skipping")

                    # remove the empty element from the list
                    self.data_files.pop(self.idx)
                    self.max_idx -= 1
                    self.idx -= 1  # we're going to add one later

                    cur_max_idx = min(cur_max_idx + 1, self.max_idx)

                if global_modelarrays is None:
                    global_modelarrays = arrays
                else:
                    global_modelarrays.concatenate(arrays)

                self.idx += 1
            if global_modelarrays is not None:

                anchor, neighbour, distant = self.find_neighbours(
                    global_modelarrays, self.neighbouring_distance, self.multiplier
                )

                if self.to_tensor:
                    anchor.to_tensor(self.device)
                    neighbour.to_tensor(self.device)
                    distant.to_tensor(self.device)
                    global_modelarrays.to_tensor(self.device)

                return (
                    anchor.return_tuple(),
                    neighbour.return_tuple(),
                    distant.return_tuple(),
                )
            else:
                raise StopIteration()

        else:  # final_x_curr >= self.max_idx
            raise StopIteration()

    @staticmethod
    def find_neighbours(
        x: ModelArrays, distance: Tuple[float, float], multiplier: int,
    ) -> Tuple[TrainData, TrainData, TrainData]:
        """
        Given a single modelArray, returns a tuple containing
        [AnchorData, NeighbourData, DistantData]
        """
        anchor_indices: List[int] = []
        neighbour_indices: List[int] = []
        distant_indices: List[int] = []

        outer_distance = tuple(multiplier * val for val in distance)

        for idx, latlon in enumerate(x.latlons):
            # to really emphasize the distance, we ensure the distant indices
            # are at least twice as far away as the neighbouring indices
            neighbours = np.where(
                (x.latlons <= latlon + distance).all(axis=1)
                & (x.latlons >= latlon - distance).all(axis=1)
            )[0]
            distants = np.where(
                (x.latlons[:, 0] > latlon[0] + outer_distance[0])
                | (x.latlons[:, 1] < latlon[1] - outer_distance[1])
                | (x.latlons[:, 0] < latlon[0] - outer_distance[0])
                | (x.latlons[:, 1] > latlon[1] + outer_distance[1])
            )[0]

            if len(distants) == 0:
                print(f"No distant values found for {latlon}")
            elif len(neighbours) == 0:
                print(f"No near values found for {latlon}")
            else:
                neighbour_indices.append(np.random.choice(neighbours))
                distant_indices.append(np.random.choice(distants))
                anchor_indices.append(idx)

        return x.x[anchor_indices], x.x[neighbour_indices], x.x[distant_indices]


def triplet_loss(
    z_anchor: torch.Tensor,
    z_neighbour: torch.Tensor,
    z_distant: torch.Tensor,
    margin: float = 0.1,
    l2: float = 0,
) -> torch.Tensor:
    """
    https://github.com/ermongroup/tile2vec/blob/master/src/resnet.py#L162
    """
    l_n = torch.sqrt(((z_anchor - z_neighbour) ** 2).sum(dim=1))
    l_d = -torch.sqrt(((z_anchor - z_distant) ** 2).sum(dim=1))
    loss = torch.mean(F.relu(l_n + l_d + margin))
    if l2 != 0:
        loss += l2 * (
            torch.norm(z_anchor) + torch.norm(z_neighbour) + torch.norm(z_distant)
        )
    return loss


def chunk_triplets(
    x1: Tuple[Optional[torch.Tensor], ...],
    x2: Tuple[Optional[torch.Tensor], ...],
    x3: Tuple[Optional[torch.Tensor], ...],
    batch_size: int,
    shuffle: bool = False,
) -> Iterable[
    Tuple[
        Tuple[Optional[torch.Tensor], ...],
        Tuple[Optional[torch.Tensor], ...],
        Tuple[Optional[torch.Tensor], ...],
    ]
]:
    """
    This function does the same thing as src.utils.models.chunk_array, but in the special case
    of tensors. We could (should?) integrate the two functions
    """
    assert (
        (x1[0] is not None) and (x2[0] is not None) and (x3[0] is not None)
    ), f"x1[0] should be historical data, and therefore should not be None"
    num_sections = max(1, x1[0].shape[0] // batch_size)

    assert (
        x1[0].shape == x2[0].shape == x3[0].shape
    ), f"All inputs must have the same number of elements!"

    split_x1 = []
    split_x2 = []
    split_x3 = []

    for idx, (x1_section, x2_section, x3_section) in enumerate(zip(x1, x2, x3)):
        if (
            (x1_section is not None)
            and (x2_section is not None)
            and (x3_section is not None)
        ):
            split_x1.append(torch.chunk(x1_section, num_sections))
            split_x2.append(torch.chunk(x2_section, num_sections))
            split_x3.append(torch.chunk(x3_section, num_sections))
        else:
            split_x1.append([None] * num_sections)  # type: ignore
            split_x2.append([None] * num_sections)  # type: ignore
            split_x3.append([None] * num_sections)  # type: ignore
    return_arrays = list(zip(*split_x1, *split_x2, *split_x3))

    chunk_lengths = len(split_x1)

    if shuffle:
        shuffle_list(return_arrays)
    return [(chunk[:chunk_lengths], chunk[chunk_lengths : 2 * chunk_lengths], chunk[2 * chunk_lengths :]) for chunk in return_arrays]  # type: ignore
