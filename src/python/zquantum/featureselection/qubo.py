import numpy as np
from typing import Union, Tuple
from dimod import BinaryQuadraticModel


def prepare_qubo_for_feature_selection(x: np.ndarray, y: np.ndarray, alpha: float):
    """Creates a QUBO based on method described in "Quadratic Programming Feature Selection" by Rodriguez et al.
    (https://www.jmlr.org/papers/volume11/rodriguez-lujan10a/rodriguez-lujan10a.pdf)

    Args:
        x: array containing input features
        y: vector containing target output
        alpha: parameter describing the relative importance of independence and relevance of features.

    Returns:
        BinaryQuadraticModel
    """
    if alpha < 0 or alpha > 1:
        raise ValueError("Wrong value of alpha parameter.")

    full_matrix = np.concatenate([x, np.array([y]).T], axis=1).T
    correlation_matrix = np.corrcoef(full_matrix)
    n_features = x.shape[1]
    Q = np.abs(correlation_matrix[:n_features, :n_features])
    F = np.abs(correlation_matrix[n_features][:-1])

    qubo_dict = {}
    for i in range(len(Q)):
        qubo_dict[(i, i)] = -1 * alpha * F[i]
        for j in range(len(Q)):
            if i != j:
                qubo_dict[(i, j)] = (1 - alpha) * Q[i][j]

    return BinaryQuadraticModel(qubo_dict, vartype="BINARY")