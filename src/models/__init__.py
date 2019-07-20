from pathlib import Path
import pickle

from typing import cast, Optional, Union

from .parsimonious import Persistence
from .regression import LinearRegression
from .neural_networks.linear_network import LinearNetwork
from .neural_networks.rnn import RecurrentNetwork

__all__ = ['Persistence', 'LinearRegression', 'LinearNetwork',
           'RecurrentNetwork']


def load_model(model_path: Path, data_path: Optional[Path] = None,
               model_type: Optional[str] = None) -> Union[RecurrentNetwork,
                                                          LinearNetwork,
                                                          LinearRegression]:
    """
    Given a path to a saved model, try and load it. If none is given,
    the function tries to infer it from the model path
    """

    str_to_model = {
        'rnn': RecurrentNetwork,
        'linear_network': LinearNetwork,
        'linear_regression': LinearRegression
    }

    # The assumption that model type is index -2 and that the data path
    # is index -5 holds true if the data folder has only been manipulated by
    # the pipeline
    if model_type is None:
        model_type = cast(str, str(model_path.parts[-2]))
    if data_path is None:
        data_path = cast(Path, model_path.parts[-5])

    with model_path.open('rb') as f:
        model_dict = pickle.load(f)

    init_kwargs = {'data_folder': data_path}
    for key, val in model_dict.items():
        if key != 'model':
            init_kwargs[key] = val

    model = str_to_model[model_type](**init_kwargs)

    model.load(**model_dict['model'])

    return model
