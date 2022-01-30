import operator
import random
from typing import Set, Tuple

import matplotlib.pyplot as plt
import perfplot

from matroids.utils.random_access_set import RandomAccessMutableSet


random.seed(2022)


SetupData = Tuple[
    int,
    Set,
    RandomAccessMutableSet,
    int,
]


def setup(n: int) -> SetupData:
    return n, set(range(n)), RandomAccessMutableSet(range(n)), random.randrange(n)


def set_remove(setup_data: SetupData):
    n, s, _, x = setup_data
    s.remove(x)
    return s


def random_access_set_remove(setup_data: SetupData):
    n, _, s, x = setup_data
    s.remove(x)
    return s


results = perfplot.bench(
    kernels=[set_remove, random_access_set_remove],
    setup=setup,
    n_range=list(range(100, 10000, 100)),
    xlabel="size",
    target_time_per_measurement=0.0,
    equality_check=operator.eq,
)
results.plot()
plt.show()
