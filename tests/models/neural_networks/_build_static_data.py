"""
functions for calculating the static embedding
"""
import torch
import numpy as np
from typing import List, Union, Tuple
from src.models.neural_networks.ealstm import EALSTM


def sigmoid(x: np.array) -> np.array:
    return 1 / (1 + np.exp(-x))


def build_static_x(
    x: Union[List[np.array], Tuple[np.array]], ealstm: EALSTM
) -> Tuple[List[np.array], List[np.array], List[np.array]]:
    all_static_x = []
    all_latlons = []
    all_pred_months = []

    for i in range(len(x)):  #  EACH BATCH (of X,y pairs)
        pred_month_data = x[i][1]
        latlons_data = x[i][2]
        yearly_aggs_data = x[i][4]
        static_data = x[i][5]

        # append the static_arrays
        static_x = []
        if ealstm.static == "features":
            if ealstm.include_latlons:
                # normalise latlon
                static_x.append(
                    (latlons_data - latlons_data.mean(axis=0))
                    / latlons_data.std(axis=0)
                )  # 0, 1
            if ealstm.include_yearly_aggs:
                static_x.append(yearly_aggs_data)  # 2: 9
            static_x.append(static_data)

            # NO LONGER INCLUDING the pred month
            if False:  #  ealstm.include_pred_month:
                # one_hot_encode the pred_month_data
                try:
                    static_x.append(
                        ealstm._one_hot(  # type: ignore
                            torch.from_numpy(pred_month_data), 12
                        ).numpy()
                    )
                except TypeError:
                    # when need to convert tensor from gpu format
                    static_x.append(
                        ealstm._one_hot(  # type: ignore
                            torch.from_numpy(pred_month_data), 12
                        )
                        .cpu()
                        .numpy()
                    )
        elif ealstm.static == "embeddings":
            assert (
                False
            ), "Have not implemented getting the embedding for pixel-id as input"

        # exclude Nones
        static_x = np.concatenate([x for x in static_x if x is not None], axis=-1)
        #  print("Static X Data Shape: ", static_x.shape)

        # all data
        all_static_x.append(static_x)

        # metadata (latlons and pred_months)
        all_latlons.append(latlons_data)
        all_pred_months.append(pred_month_data)

    return all_static_x, all_latlons, all_pred_months


def calculate_embeddings(static_x: np.ndarray, W: np.ndarray, b: np.array) -> np.array:
    assert (
        W.T.shape[0] == static_x.shape[-1]
    ), f"Matrix operations must be valid {static_x.shape} * {W.T.shape}"

    embedding = []
    for pixel_ix in range(static_x.shape[0]):
        embedding.append(sigmoid(np.dot(W, static_x[pixel_ix]) + b))
    return np.array(embedding)


def get_static_embedding(
    ealstm,
) -> Tuple[List[np.array], Tuple[List[np.array], np.array, List[np.array]]]:
    # get W, b from state_dict
    od = ealstm.model.static_embedding.state_dict()
    try:
        W = od["weight"].numpy()
        b = od["bias"].numpy()
    except TypeError:
        W = od["weight"].cpu().numpy()
        b = od["bias"].cpu().numpy()

    # get X_static data from dataloader
    print("Calling Training DataLoader")
    dl = ealstm.get_dataloader("train", batch_file_size=1, shuffle_data=False)
    x = [x for (x, y) in dl]

    # build static_x matrix
    all_static_x, all_latlons, all_pred_months = build_static_x(x, ealstm=ealstm)
    # check w^Tx + b is a valid matrix operation
    assert (
        W.T.shape[0] == all_static_x[0].shape[-1]
    ), f"W.T shape: {W.T.shape} static_x shape: {all_static_x[0].shape}"

    # calculate the embeddings
    all_embeddings = []
    for static_x in all_static_x:
        embedding = calculate_embeddings(static_x, W=W, b=b)
        all_embeddings.append(embedding)

    return (
        all_embeddings,
        (all_static_x, np.array(all_latlons), np.array(all_pred_months)),
    )
