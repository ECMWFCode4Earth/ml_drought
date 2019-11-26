import numpy as np
import torch

from src.models.utils import chunk_array


class TestChunker:
    def test_chunk_array(self):

        test_array = np.ones((10, 2, 1))

        for x, y in chunk_array(test_array, test_array, 2, shuffle=True):
            assert (x[0].shape == (2, 2, 1)) and (y.shape == (2, 2, 1))

    def test_chunk_tensor(self):

        test_array = torch.ones(10, 2, 1)

        for x, y in chunk_array(test_array, test_array, 2, shuffle=True):
            assert (x[0].shape == (2, 2, 1)) and (y.shape == (2, 2, 1))

    def test_shuffling_array(self):
        test_x = np.arange(0, 10)
        test_y = np.arange(0, 10)

        for x, y in chunk_array(test_x, test_y, 2, shuffle=True):
            assert (x[0] == y).all()

    def test_shuffling_tensor(self):
        test_x = torch.arange(0, 10)
        test_y = torch.arange(0, 10)

        for x, y in chunk_array(test_x, test_y, 2, shuffle=True):
            assert (x[0] == y).all()

    def test_tuple_input(self):
        test_x = torch.arange(0, 10)
        test_y = torch.arange(0, 10)

        for x, y in chunk_array((test_x,), test_y, 2, shuffle=True):
            assert (x[0] == y).all()

    def test_chunk_multiple(self):
        test_x_1 = torch.arange(0, 10)
        test_x_2 = torch.arange(0, 10)
        test_y = torch.arange(0, 10)

        for x, y in chunk_array((test_x_1, test_x_2), test_y, 2, shuffle=True):
            assert (x[0] == x[1]).all()
            assert (x[0] == y).all()

    def test_tensor_with_nones(self):
        test_x_1 = torch.arange(0, 10)
        test_x_2 = None
        test_x_3 = torch.arange(0, 10)
        test_y = torch.arange(0, 10)

        for x, y in chunk_array(
            (test_x_1, test_x_2, test_x_3), test_y, 2, shuffle=True
        ):
            assert (x[0] == x[2]).all()
            assert (x[0] == y).all()

    def test_numpy_with_nones(self):
        test_x_1 = np.arange(0, 10)
        test_x_2 = None
        test_x_3 = np.arange(0, 10)
        test_y = np.arange(0, 10)

        for x, y in chunk_array(
            (test_x_1, test_x_2, test_x_3), test_y, 2, shuffle=True
        ):
            assert (x[0] == x[2]).all()
            assert (x[0] == y).all()
