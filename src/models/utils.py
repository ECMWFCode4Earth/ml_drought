import torch
import numpy as np
from random import shuffle as shuffle_list
from typing import Iterable, Union, Tuple


def chunk_array(x: Union[torch.Tensor, np.ndarray],
                y: Union[torch.Tensor, np.ndarray],
                batch_size: int,
                shuffle: bool = False) -> Iterable[Tuple[Union[torch.Tensor, np.ndarray],
                                                         Union[torch.Tensor, np.ndarray]]]:
    num_sections = max(1, x.shape[0] // batch_size)
    if type(x) == np.ndarray:
        return _chunk_ndarray(x, y, num_sections, shuffle)
    else:
        return _chunk_tensor(x, y, num_sections, shuffle)


def _chunk_ndarray(x: np.ndarray, y: np.ndarray,
                   num_sections: int,
                   shuffle: bool) -> Iterable[Tuple[np.ndarray, np.ndarray]]:

    split_x, split_y = np.array_split(x, num_sections), np.array_split(y, num_sections)

    if shuffle:
        shuffle_list(split_x)
        shuffle_list(split_y)
    return zip(split_x, split_y)


def _chunk_tensor(x: torch.Tensor, y: torch.Tensor,
                  num_sections: int,
                  shuffle: bool) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    split_x = torch.chunk(x, num_sections)
    split_y = torch.chunk(y, num_sections)
    if shuffle:
        shuffle_list(list(split_x))
        shuffle_list(list(split_y))

    return zip(split_x, split_y)
