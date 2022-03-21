import time
import typing as tp


class Stopwatch(tp.ContextManager["Stopwatch"]):
    """
    Context manager for simple performance measurement.

    Usage:

        with Stopwatch() as stopwatch:
            <some code>

        print(stopwatch.measurement)

    """

    def __init__(self):
        self.start_time = self.end_time = None

    @property
    def measurement(self) -> float:
        if self.end_time is None:
            raise ValueError("No measurement has been performed yet")
        return self.end_time - self.start_time

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
