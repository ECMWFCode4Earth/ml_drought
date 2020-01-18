import torch

from src.models.neural_networks.triplet_data import chunk_triplets


class TestChunker:
    def test_chunk_tensor(self):

        test_array = torch.ones(10, 2, 1)

        for x1, x2, x3 in chunk_triplets(
            (test_array,), (test_array,), (test_array,), 2, shuffle=True
        ):
            assert (
                (x1[0].shape == (2, 2, 1))
                and (x2[0].shape == (2, 2, 1))
                and (x3[0].shape == (2, 2, 1))
            )

    def test_shuffling_tensor(self):
        test_x1 = torch.arange(0, 10)
        test_x2 = torch.arange(0, 10)
        test_x3 = torch.arange(0, 10)

        for x1, x2, x3 in chunk_triplets(
            (test_x1,), (test_x2,), (test_x3,), 2, shuffle=True
        ):
            assert (x1[0] == x2[0]).all()

    def test_tensor_with_nones(self):
        test_x_1 = torch.arange(0, 10)
        test_x_2 = None
        test_x_3 = torch.arange(0, 10)

        for x1, x2, x3 in chunk_triplets(
            (test_x_1, test_x_2, test_x_3),
            (test_x_1, test_x_2, test_x_3),
            (test_x_1, test_x_2, test_x_3),
            2,
            shuffle=True,
        ):
            assert (x1[0] == x1[2]).all()
            assert (x2[0] == x2[2]).all()
            assert (x3[0] == x3[2]).all()
