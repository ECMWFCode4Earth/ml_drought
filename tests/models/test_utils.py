import numpy as np
import torch

from src.models.utils import chunk_array


class TestChunker:

    def test_chunk_array(self):

        test_array = np.ones((10, 2, 1))

        for x, y in chunk_array(test_array, test_array, 2, shuffle=True):
            assert ((x.shape == (2, 2, 1)) and (y.shape == (2, 2, 1)))

    def test_chunk_tensor(self):

        test_array = torch.ones(10, 2, 1)

        for x, y in chunk_array(test_array, test_array, 2, shuffle=True):
            assert ((x.shape == (2, 2, 1)) and (y.shape == (2, 2, 1)))
