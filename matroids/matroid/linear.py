import dataclasses
import typing
import numpy as np

from . import Matroid, WeightedMatroid


@dataclasses.dataclass(frozen=True)
class RealLinearMatroid(Matroid[int]):
    """
    A linear matroid for vectors in R^n.

    A linear matroid is one that can be represented by a matrix. The ground set E is
    a set of vectors in a vector space V over a field F. In this case, F is the field
    of real numbers. The family of independent sets is contains all sets of linearly
    independent vectors made up from elements of E. The matrix representation is
    obtained by writing each vector in E as a column.

    This type of matroid is implemented here by identifying each vector with an integer
    index, corresponding to its index in the matrix. The matrix is stored as a
    (read-only) property of the object.
    """

    matrix: np.ndarray  #: matrix of real vectors

    def __post_init__(self):
        # validate matrix
        if not self.matrix.ndim == 2:
            raise ValueError(
                f"Given array is not a matrix: has {self.matrix.ndim} dimensions"
            )

    @property
    def ground_set(self) -> typing.Set[int]:
        # return the indices of columns in the matrix
        return set(range(self.matrix.shape[1]))

    def is_independent(self, subset: typing.Set[int]) -> bool:
        # fetch the given columns and check whether the resulting matrix is full-rank
        columns_subset = self.get_matrix(subset)
        return np.linalg.matrix_rank(columns_subset) == columns_subset.shape[1]

    def get_matrix(self, subset: typing.Set[int]) -> np.ndarray:
        """
        Return the sub-matrix corresponding to the given subset of elements (columns).

        :param subset: Subset of column indices of this matroid's matrix.
        :return: The sub-matrix corresponding to the given subset,
             preserving column order.
        """
        return self.matrix[:, sorted(subset)]

@dataclasses.dataclass(frozen=True)
class WeightedRealLinearMatroid(RealLinearMatroid, WeightedMatroid):
    weights: np.ndarray  #: vector of weights

    def __init__(self, matrix: np.ndarray, weights: np.ndarray = None):
        super().__init__(matrix)

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
        object.__setattr__(self, "weights", weights)

    def get_weight(self, element: int) -> float:
        return self.weights[element]
