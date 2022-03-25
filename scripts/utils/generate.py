import random

from matroids.matroid import MutableIntUniformMatroid, MutableMatroid


def generate_dummy_matroid(
    size: int, rank: int, uniform_weights: bool
) -> MutableMatroid:
    weights = {} if uniform_weights else {i: random.random() for i in range(size)}
    return MutableIntUniformMatroid(size, rank, weights)
