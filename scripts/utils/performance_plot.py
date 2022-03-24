import dataclasses
import functools
import operator
import typing as tp

import matplotlib.pyplot as plt
import matplotlib.style
import matplotx.styles
import numpy as np
import tqdm

from utils.save import save_figure


MATPLOTLIB_STYLE = matplotx.styles.dufte | {
    "font.size": 10,
    "axes.titlesize": 12,
}
matplotlib.style.use(MATPLOTLIB_STYLE)


PerformanceMeasurements = tp.Dict[str, np.ndarray]


@dataclasses.dataclass
class PerformanceExperiment:
    """
    An experiment for measuring the execution time of a set of procedures.

    - title: Title of the experiment.
    - timer_functions: The timer functions to use. The keys are labels for the
        procedures being evaluated, and the values are timer functions, each of which
        should measure the execution time of the procedure given some input data
        and return the time elapsed in seconds as a float.
    - x_name: Name of the dependent variable being changed (x). Used as the
        parameter name for passing the value to each of the timer functions.
    - x_range: Range of values for the dependent variable being changed.
    - fixed_variables: Keyword arguments passed to each timer function for each
        repetition. Represents dependent variables that remain fixed.
    - repeats: Number of measurement repetitions per procedure and x value.
    """

    title: str

    timer_functions: tp.Dict[str, tp.Callable[..., float]]
    x_name: str
    x_range: tp.Sequence
    fixed_variables: tp.Dict[str, tp.Any]
    repeats: int = 5

    def measure_performance(self) -> PerformanceMeasurements:
        """Run the experiment (gather measurements of execution time)."""
        results = {
            label: np.full((len(self.x_range), self.repeats), fill_value=np.nan)
            for label in self.timer_functions
        }

        for j, x_value in enumerate(
            tqdm.tqdm(self.x_range, desc=self.title.replace("\n", " / "))
        ):
            input_variables = self.fixed_variables | {self.x_name: x_value}
            for label, timer in self.timer_functions.items():
                times = results[label]
                for k in range(self.repeats):
                    times[j, k] = timer(**input_variables)

        return results

    def plot_performance(
        self,
        ax: plt.Axes,
        measurements: PerformanceMeasurements,
    ) -> None:
        """
        Plot performance measurements obtained from :meth:`measure_performance`.

        :param ax: Axes object on which to plot.
        :param measurements: Measurements to plot, obtained from
            :meth:`measure_performance`.
        """
        if self.title is not None:
            ax.set_title(self.title)
        ax.set_xlabel(self.x_name)
        ax.set_ylabel("time (s)")

        for label, times in measurements.items():
            means = times.mean(axis=-1)
            stds = times.std(axis=-1)
            ax.errorbar(self.x_range, means, yerr=stds, marker=".", label=label)

        ax.set_ylim(bottom=0)
        ax.ticklabel_format(axis="y", scilimits=(-2, 2))

    def measure_and_plot(self, ax: plt.Axes) -> None:
        """
        Combines measurement and plotting in one step.

        :param ax: Axes object on which to plot.
        """
        self.plot_performance(ax, self.measure_performance())


@dataclasses.dataclass
class PerformanceExperimentGroup:
    title: str

    experiments: tp.Sequence[PerformanceExperiment]

    def measure_performance(self) -> tp.Tuple[PerformanceMeasurements, ...]:
        """Run each of the experiments in the group."""
        return tuple(e.measure_performance() for e in self.experiments)

    def plot_performance(
        self, all_measurements: tp.Sequence[PerformanceMeasurements]
    ) -> plt.Figure:
        """Creates a new figure and plots the measurements of each experiment."""
        fig, axes = plt.subplots(
            nrows=1,
            ncols=len(self.experiments),
            squeeze=False,
            sharey="row",
            figsize=[2 + 4 * len(self.experiments), 4.8],
        )
        fig: plt.Figure
        fig.suptitle(self.title)

        for experiment, measurement, ax in zip(
            self.experiments, all_measurements, axes.flat
        ):
            experiment.plot_performance(ax, measurement)

        # legend - avoid duplicate labels
        artist_dicts = [
            dict(zip(labels, handles))
            for handles, labels in map(plt.Axes.get_legend_handles_labels, axes.flat)
        ]
        artist_dict = functools.reduce(operator.or_, artist_dicts)  # dict union
        fig.legend(
            artist_dict.values(),
            artist_dict.keys(),
            loc="upper left",
            bbox_to_anchor=(1.0, 1.0),
        )

        fig.tight_layout()
        return fig

    def show_performance(
        self, all_measurements: tp.Sequence[PerformanceMeasurements]
    ) -> None:
        fig = self.plot_performance(all_measurements)
        plt.show()
        save_figure(fig, identifiers=[self.title])

    def measure_and_show(self) -> None:
        """Shortcut for running the experiments and showing the plot in one step."""
        self.show_performance(self.measure_performance())
