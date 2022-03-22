import dataclasses
import functools
import typing as tp

import numpy as np

from .base import Matroid


@dataclasses.dataclass(eq=False)
class RealLinearMatroid(Matroid[int]):
    """
    A linear matroid for vectors in R^n.

    A linear matroid is one that can be represented by a matrix. The ground set E is
    the set of column indices of a matrix M over a field F. In this case, F is the field
    of real numbers. The family of independent sets is contains all sets of linearly
    independent columns of M.

    This implementation uses 0-based integer indices as the column indices (i.e. the
    elements of the ground set). The matrix is stored as a (read-only) property of
    the object.

    Weights are stored as an n-dimensional real vector, where n is the number of
    columns.
    """

    matrix: np.ndarray  #: matrix of real vectors
    weights: np.ndarray  #: vector of weights

    def __init__(self, matrix: np.ndarray, weights: np.ndarray = None):
        # validate matrix
        if not matrix.ndim == 2:
            raise ValueError(
                f"Given array is not a matrix: has {self.matrix.ndim} dimensions"
            )
        matrix = matrix.astype(float)

        # validate and store weights
        weights_shape = (matrix.shape[1],)
        if weights is not None:
            if weights.shape != weights_shape:
                raise ValueError(
                    f"Invalid weights vector: has shape {weights.shape!r},"
                    f" expected {weights_shape!r}"
                )
            weights = weights.astype(float)
        else:
            weights = np.ones(weights_shape, dtype=float)

        # must use setattr manually since this is a frozen dataclass
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "weights", weights)

    @property
    @functools.cache  # this matroid is not mutable so we can memoize
    def ground_set(self) -> tp.AbstractSet[int]:
        # return the indices of columns in the matrix
        return set(range(self.matrix.shape[1]))

    def __bool__(self):
        return bool(self.matrix.shape[1])

    def is_independent(self, subset: tp.AbstractSet[int]) -> bool:
        # fetch the given columns and check whether the resulting matrix is full-rank
        columns_subset = self.get_matrix(subset)
        # shortcut if the number of vectors is greater than the dimension of R^n
        if columns_subset.shape[1] > columns_subset.shape[0]:
            return False
        return np.linalg.matrix_rank(columns_subset) == columns_subset.shape[1]

    def get_weight(self, element: int) -> float:
        return self.weights[element]

    def get_matrix(self, subset: tp.AbstractSet[int]) -> np.ndarray:
        """
        Return the sub-matrix corresponding to the given subset of elements (columns).

        :param subset: Subset of column indices of this matroid's matrix.
        :return: The sub-matrix corresponding to the given subset,
             preserving column order.
        """
        return self.matrix[:, sorted(subset)]
