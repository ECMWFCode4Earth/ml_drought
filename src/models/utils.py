import torch
import numpy as np
from typing import Iterable, Union, Tuple


def chunk_array(x: Union[torch.Tensor, np.ndarray],
                y: Union[torch.Tensor, np.ndarray],
                batch_size: int) -> Iterable[Tuple[Union[torch.Tensor, np.ndarray],
                                                   Union[torch.Tensor, np.ndarray]]]:

    if type(x) == np.ndarray:
        return _chunk_ndarray(x, y, batch_size)
    else:
        return _chunk_tensor(x, y, batch_size)


def _chunk_ndarray(x: np.ndarray, y: np.ndarray,
                   batch_size: int) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    num_sections = x.shape[0] // batch_size

    return zip(np.array_split(x, num_sections), np.array_split(y, num_sections))


def _chunk_tensor(x: torch.Tensor, y: torch.Tensor,
                  batch_size: int) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    num_sections = x.shape[0] // batch_size

    return zip(torch.chunk(x, num_sections), torch.chunk(y, num_sections))
