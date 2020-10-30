from typing import Dict, Tuple, List

import numpy as np
from scipy.stats import wilcoxon


def ecdf(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate empirical cummulative density function

    Parameters
    ----------
    x : np.ndarray
        Array containing the data

    Returns
    -------
    x : np.ndarray
        Array containing the sorted metric values
    y : np.ndarray]
        Array containing the sorted cdf values
    """
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / float(len(xs))
    return xs, ys


def get_pvals(metrics: dict, model1: str, model2: str) -> Tuple[List, float]:
    """[summary]

    Parameters
    ----------
    metrics : dict
        Dictionary, containing the metric values of both models for all basins.
    model1 : str
        String, defining the first model to take. Must be a key in `metrics`
    model2 : str
        String, defining the second model to take. Must be a key in `metrics`

    Returns
    -------
    p_vals : List
        List, containing the p-values of all possible seed combinations.
    p_val : float
        P-value between the ensemble means.
    """

    # p-values between mean performance per basin of both models
    metric_model1 = get_mean_basin_performance(metrics, model1)
    metric_model2 = get_mean_basin_performance(metrics, model2)
    _, p_val_single = wilcoxon(
        list(metric_model1.values()), list(metric_model2.values())
    )

    # p-value between ensemble means
    _, p_val_ensemble = wilcoxon(
        list(metrics[model1]["ensemble"].values()),
        list(metrics[model2]["ensemble"].values()),
    )
    return p_val_single, p_val_ensemble


def get_cohens_d(values1: List, values2: List) -> float:
    """Calculate Cohen's Effect size

    Parameters
    ----------
    values1 : List
        List of model performances of model 1
    values2 : List
        List of model performances of model 2

    Returns
    -------
    float
        Cohen's d
    """
    s = np.sqrt(
        ((len(values1) - 1) * np.var(values1) + (len(values2) - 1) * np.var(values2))
        / (len(values1) + len(values2) - 2)
    )
    d = (np.abs(np.mean(values1) - np.mean(values2))) / s
    return d
