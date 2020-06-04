import torch
import numpy as np
import pandas as pd
from random import shuffle as shuffle_list
from typing import cast, Iterable, Union, Tuple, Optional


def chunk_array(
    x: Union[
        Tuple[Union[torch.Tensor, np.ndarray, None], ...],
        Union[torch.Tensor, np.ndarray],
    ],
    y: Union[torch.Tensor, np.ndarray],
    batch_size: int,
    shuffle: bool = False,
) -> Iterable[
    Tuple[
        Tuple[Union[torch.Tensor, np.ndarray, None], ...],
        Union[torch.Tensor, np.ndarray],
    ]
]:
    """
    Chunk an array into batches of batch size `batch_size`

    Arguments
    ----------
    x: ({torch.Tensor, np.ndarray})
        The x tensors to chunk
    y: {torch.Tensor, np.ndarray}
        The y tensor to chunk. Must be the same type as x
    batch_size: int
        The size of the batches to return
    shuffle: bool = False
        Whether to shuffle the returned tensors

    Returns
    ----------
    An iterator returning tuples of batches (x, y)
    """
    if type(x) is not tuple:
        x = (x,)
    x = cast(Tuple[Union[torch.Tensor, np.ndarray, None], ...], x)

    assert (
        x[0] is not None
    ), f"x[0] should be historical data, and therefore should not be None"
    num_sections = max(1, x[0].shape[0] // batch_size)

    if type(x[0]) == np.ndarray:
        return _chunk_ndarray(x, y, num_sections, shuffle)
    else:
        return _chunk_tensor(x, y, num_sections, shuffle)


def _chunk_ndarray(
    x: Tuple[Optional[np.ndarray], ...], y: np.ndarray, num_sections: int, shuffle: bool
) -> Iterable[Tuple[Tuple[Optional[np.ndarray], ...], np.ndarray]]:

    split_x = []
    for idx, x_section in enumerate(x):
        if x_section is not None:
            split_x.append(np.array_split(x_section, num_sections))
        else:
            split_x.append([None] * num_sections)
    split_y = np.array_split(y, num_sections)
    return_arrays = list(zip(*split_x, split_y))

    if shuffle:
        shuffle_list(return_arrays)
    return [(chunk[:-1], chunk[-1]) for chunk in return_arrays]  # type: ignore


def _chunk_tensor(
    x: Tuple[Optional[torch.Tensor], ...],
    y: torch.Tensor,
    num_sections: int,
    shuffle: bool,
) -> Iterable[Tuple[Tuple[Optional[torch.Tensor], ...], torch.Tensor]]:
    split_x = []

    for idx, x_section in enumerate(x):
        if x_section is not None:
            split_x.append(torch.chunk(x_section, num_sections))
        else:
            split_x.append([None] * num_sections)  # type: ignore
    split_y = torch.chunk(y, num_sections)
    return_arrays = list(zip(*split_x, split_y))
    assert False

    if shuffle:
        shuffle_list(return_arrays)
    return [(chunk[:-1], chunk[-1]) for chunk in return_arrays]  # type: ignore


def _datetime_to_folder_time_str(date: np.datetime64) -> str:
    date = pd.to_datetime(date)
    return f"{str(date.year[0])}_{str(date.month[0])}"
