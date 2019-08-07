from pathlib import Path
import pickle

from typing import cast, Optional, Union

from .parsimonious import Persistence
from .regression import LinearRegression
from .neural_networks.linear_network import LinearNetwork
from .neural_networks.rnn import RecurrentNetwork
from .neural_networks.ealstm import EARecurrentNetwork

__all__ = ['Persistence', 'LinearRegression', 'LinearNetwork',
           'RecurrentNetwork', 'EARecurrentNetwork']


def load_model(model_path: Path, data_path: Optional[Path] = None,
               model_type: Optional[str] = None) -> Union[RecurrentNetwork,
                                                          LinearNetwork,
                                                          LinearRegression]:
    """
    This function loads models from the output `.pkl` files generated when
    calling model.save()

    Arguments
    ----------
    model_path: Path
        The path to the model
    data_path: Optional[Path] = None
        The path to the data folder. If None, the function infers this from the
        model_path (assuming it was saved as part of the pipeline)
    model_type: Optional[str] = None
        The type of model to load. If None, the function infers this from the
        model_path (assuming it was saved as part of the pipeline)

    Returns
    ----------
    model: Union[RecurrentNetwork, LinearNetwork, LinearRegression]
        A model object loaded from the model_path
    """

    str_to_model = {
        'rnn': RecurrentNetwork,
        'linear_network': LinearNetwork,
        'linear_regression': LinearRegression,
        'ealstm': EARecurrentNetwork
    }

    # The assumption that model type is index -2 and that the data path
    # is index -5 holds true if the data folder has only been manipulated by
    # the pipeline
    if model_type is None:
        model_type = cast(str, str(model_path.parts[-2]))
    if data_path is None:
        data_path = cast(Path, model_path.parents[3])

    with model_path.open('rb') as f:
        model_dict = pickle.load(f)

    init_kwargs = {'data_folder': data_path}
    for key, val in model_dict.items():
        if key != 'model':
            init_kwargs[key] = val
    print(init_kwargs)
    model = str_to_model[model_type](**init_kwargs)

    model.load(**model_dict['model'])

    return model
