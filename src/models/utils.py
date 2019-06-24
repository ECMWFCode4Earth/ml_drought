import torch
import numpy as np
from random import shuffle as shuffle_list
from typing import Iterable, Union, Tuple


def chunk_array(x: Union[torch.Tensor, np.ndarray],
                y: Union[torch.Tensor, np.ndarray],
                batch_size: int,
                shuffle: bool = False) -> Iterable[Tuple[Union[torch.Tensor, np.ndarray],
                                                         Union[torch.Tensor, np.ndarray]]]:
    """
    Chunk an array into batches of batch size `batch_size`

    Arguments
    ----------
    x: {torch.Tensor, np.ndarray}
        The x tensor to chunk
    y: {torch.Tensor, np.ndarray}
        The y tensor to chunk. Must be the same type as x
    batch_size: int
        The size of the batches to return
    shuffle: bool = False
        Whether to shuffle the returned tensors
    """
    num_sections = max(1, x.shape[0] // batch_size)
    if type(x) == np.ndarray:
        return _chunk_ndarray(x, y, num_sections, shuffle)
    else:
        return _chunk_tensor(x, y, num_sections, shuffle)


def _chunk_ndarray(x: np.ndarray, y: np.ndarray,
                   num_sections: int,
                   shuffle: bool) -> Iterable[Tuple[np.ndarray, np.ndarray]]:

    split_x, split_y = np.array_split(x, num_sections), np.array_split(y, num_sections)
    return_arrays = list(zip(split_x, split_y))

    if shuffle:
        shuffle_list(return_arrays)
    return return_arrays


def _chunk_tensor(x: torch.Tensor, y: torch.Tensor,
                  num_sections: int,
                  shuffle: bool) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    split_x, split_y = torch.chunk(x, num_sections), torch.chunk(y, num_sections)
    return_arrays = list(zip(split_x, split_y))

    if shuffle:
        shuffle_list(return_arrays)
    return return_arrays

