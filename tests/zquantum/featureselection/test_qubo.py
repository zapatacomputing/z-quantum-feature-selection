import numpy as np
from zquantum.featureselection.qubo import prepare_qubo_for_feature_selection
import pytest


@pytest.mark.parametrize(
    "alpha,target_linear,target_quadratic,target_offset",
    [
        [
            0,
            {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
            {
                (0, 1): 2.0,
                (0, 2): 2.0,
                (0, 3): 2.0,
                (1, 2): 2.0,
                (1, 3): 2.0,
                (2, 3): 2.0,
            },
            0,
        ],
        [
            0.5,
            {0: -0.5, 1: -0.5, 2: -0.5, 3: -0.5},
            {
                (0, 1): 1.0,
                (0, 2): 1.0,
                (0, 3): 1.0,
                (1, 2): 1.0,
                (1, 3): 1.0,
                (2, 3): 1.0,
            },
            0,
        ],
        [
            1.0,
            {0: -1, 1: -1, 2: -1, 3: -1},
            {
                (0, 1): 0.0,
                (0, 2): 0.0,
                (0, 3): 0.0,
                (1, 2): 0.0,
                (1, 3): 0.0,
                (2, 3): 0.0,
            },
            0,
        ],
    ],
)
def test_prepare_qubo_for_feature_selection_gives_correct_results(
    alpha, target_linear, target_quadratic, target_offset
):
    x = np.array([[1, 2, 3, 4], [2, 4, 6, 8]])
    y = np.array([1, 2])

    qubo = prepare_qubo_for_feature_selection(x, y, alpha=alpha)

    assert len(qubo.linear) == len(target_linear)
    assert len(qubo.quadratic) == len(target_quadratic)

    assert np.isclose(qubo.offset, target_offset)
    assert str(qubo.vartype) == "Vartype.BINARY"

    for key in qubo.linear.keys():
        assert np.isclose(qubo.linear[key], target_linear[key])

    for key in qubo.quadratic.keys():
        assert np.isclose(qubo.quadratic[key], target_quadratic[key])


@pytest.mark.parametrize("alpha", [-1, 2])
def test_prepare_qubo_for_feature_selection_throws_exception_for_invalid_values_of_alpha(
    alpha,
):
    x = np.array([[1, 2, 3, 4], [2, 4, 6, 8]])
    y = np.array([1, 2])

    with pytest.raises(ValueError):
        prepare_qubo_for_feature_selection(x, y, alpha=alpha)