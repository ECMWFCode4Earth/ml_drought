from pathlib import Path
import numpy as np

from ..data import DataLoader, _BaseIter, ModelArrays, TrainData

from typing import Union, Tuple, Optional, List


class TripletLoader(DataLoader):
    def __init__(
        self,
        neighbouring_distance: Union[float, Tuple[float, float]],
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
        if isinstance(neighbouring_distance, float):
            neighbouring_distance = (neighbouring_distance, neighbouring_distance)
        self.neighbouring_distance = neighbouring_distance

    def __iter__(self):
        return _TripletIter(self)

    def __len__(self) -> int:
        return len(self.data_files) // self.batch_file_size


class _TripletIter(_BaseIter):
    def __init__(self, loader: TripletLoader) -> None:
        super().__init__(loader)
        self.neighbouring_distance = loader.neighbouring_distance

    def __next__(self) -> Tuple[TrainData, TrainData, TrainData]:

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
                    global_modelarrays, self.neighbouring_distance
                )

                if self.to_tensor:
                    anchor.to_tensor(self.device)
                    neighbour.to_tensor(self.device)
                    distant.to_tensor(self.device)
                    global_modelarrays.to_tensor(self.device)

                return anchor, neighbour, distant
            else:
                raise StopIteration()

        else:  # final_x_curr >= self.max_idx
            raise StopIteration()

    @staticmethod
    def find_neighbours(
        x: ModelArrays, distance: Tuple[float, float]
    ) -> Tuple[TrainData, TrainData, TrainData]:
        """
        Given a single modelArray, returns a tuple containing
        [AnchorData, NeighbourData, DistantData]
        """

        neighbour_indices: List[int] = []
        distant_indices: List[int] = []

        for latlon in x.latlons:
            # to really emphasize the distance, we ensure the distant indices
            # are at least twice as far away as the neighbouring indices
            neighbours = np.where(
                (x.latlons <= latlon + distance).all(axis=1)
                & (x.latlons >= latlon - distance).all(axis=1)
            )[0]
            distants = np.where(
                (x.latlons > latlon + 2 * distance).all(axis=1)
                | (x.latlons < latlon - 2 * distance).all(axis=1)
            )[0]

            neighbour_indices.append(np.random.choice(neighbours))
            distant_indices.append(np.random.choice(distants))

        return x.x, x.x[neighbour_indices], x.x[distant_indices]
