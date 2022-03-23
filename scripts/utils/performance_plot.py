import dataclasses
import typing as tp

import matplotlib.pyplot as plt
import numpy as np
import tqdm


PerformanceMeasurements = tp.Dict[str, np.ndarray]


@dataclasses.dataclass
class PerformanceExperiment:
    """
    An experiment for measuring the execution time of a set of procedures.

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
    - title: Title of the experiment (optional).
    """

    timer_functions: tp.Dict[str, tp.Callable[..., float]]
    x_name: str
    x_range: tp.Sequence
    fixed_variables: tp.Dict[str, tp.Any]
    repeats: int = 5

    title: tp.Optional[str] = None

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
            plt.errorbar(self.x_range, means, yerr=stds, marker=".", label=label)

        ax.set_ylim(bottom=0)
        ax.legend()

    def measure_and_plot(self, ax: plt.Axes) -> None:
        """
        Combines measurement and plotting in one step.

        :param ax: Axes object on which to plot.
        """
        self.plot_performance(ax, self.measure_performance())

    def measure_and_show(self) -> None:
        """Shortcut for creating the figure and showing it afterwards."""
        fig, ax = plt.subplots()
        self.measure_and_plot(ax)
        fig.tight_layout()
        plt.show()
