import typing as tp

import matplotlib.pyplot as plt
import numpy as np


def plot_mean_and_range(
    ax: plt.Axes, x: tp.Sequence, y: np.ndarray, *, label: str, **kwargs
) -> None:
    """
    Plot the mean of a series together with its range.

    The mean is plotted as a regular line plot, while the range as color filled between
    the minimum and the maximum.

    :param ax: Axis in which to plot.
    :param x: 1-axis array of x-axis values.
    :param y: Array of y-axis values. Can have any number of axes. The mean and range
        is computed for each index of the first axis.
    :param label: Label with which to plot.
    :param kwargs: Additional keyword arguments passed to :meth:`plt.Axes.plot`.
    """
    axes = tuple(range(len(y.shape)))
    mean = y.mean(axis=axes[1:])
    minimum = np.min(y, axis=axes[1:])
    maximum = np.max(y, axis=axes[1:])

    lines = ax.plot(x, mean, label=f"{label} (mean)", **kwargs)
    color = lines[-1].get_color()
    ax.fill_between(
        x, minimum, maximum, label=f"{label} (range)", color=color, alpha=0.15
    )
