from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt

import numpy as np

from typing import Dict, List


def plot_shap_values(x: np.ndarray,
                     shap_values: np.ndarray,
                     val_list: List[str],
                     normalizing_dict: Dict[str, Dict[str, float]],
                     value_to_plot: str,
                     normalize_shap_plots: bool = True,
                     show: bool = False) -> None:
    """Plots the denormalized values against their shap values, so that
    variations in the input features to the model can be compared to their effect
    on the model. For example plots, see notebooks/08_gt_recurrent_model.ipynb.
    Parameters:
    ----------
    x: np.array
        The input to a model for a single data instance
    shap_values: np.array
        The corresponding shap values (to x)
    val_list: list
        A list of the variable names, for axis labels
    normalizing_dict: dict
        The normalizing dict saved by the `Engineer`, so that the x array can be
        denormalized
    value_to_plot: str
        The specific input variable to plot. Must be in val_list
    normalize_shap_plots: bool = True
        If True, then the scale of the shap plots will be uniform across all
        variable plots (on an instance specific basis).
    show: bool = False
        If True, a plot of the variable `value_to_plot` against its shap values will be plotted.
    """
    # first, lets isolate the lists
    idx = val_list.index(value_to_plot)

    x_val = x[:, idx]

    # we also want to denormalize
    for norm_var in normalizing_dict.keys():
        if value_to_plot.endswith(norm_var):
            x_val = (x_val * normalizing_dict[norm_var]['std']) + \
                normalizing_dict[norm_var]['mean']
            break

    shap_val = shap_values[:, idx]

    months = list(range(1, len(x_val) + 1))

    host = host_subplot(111, axes_class=AA.Axes)
    plt.subplots_adjust(right=0.75)

    par1 = host.twinx()
    par1.axis["right"].toggle(all=True)

    if normalize_shap_plots:
        par1.set_ylim(shap_values.min(), shap_values.max())

    host.set_xlabel("Months")
    host.set_ylabel(value_to_plot)
    par1.set_ylabel("Shap value")

    p1, = host.plot(months, x_val, label=value_to_plot)
    p2, = par1.plot(months, shap_val, label="shap value")

    host.axis["left"].label.set_color(p1.get_color())
    par1.axis["right"].label.set_color(p2.get_color())

    host.legend()

    plt.draw()
    if show:
        plt.show()
