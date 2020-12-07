import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

from zquantum.core.utils import (
    SCHEMA_VERSION,
    convert_array_to_dict,
    convert_dict_to_array,
    sample_from_probability_distribution,
    convert_bitstrings_to_tuples,
    convert_tuples_to_bitstrings,
    save_list,
    load_list,
)

from zquantum.featureselection import (
                               quadratic_programming_feature_selection,
                               generate_reduced_quadratic_program_with_qpfs,
                               greedy_mrmr_feature_selection,)


def save_quadratic_program(quadratic_terms, linear_terms, filename, constant_term=np.zeros(1)):
    """Saves an unconstrained quadratic program specification to a JSON file.

    Args:
        quadratic_terms (np.ndarray): quadratic terms matrix
        linear_terms (np.ndarray): linear terms vector
        filename (string): name of the output file
    """
    f = open(filename, "w")
    qp_dict = {"quadratic_terms": convert_array_to_dict(quadratic_terms), 
            "linear_terms": convert_array_to_dict(linear_terms), 
            "constant_term": convert_array_to_dict(constant_term), }
    qp_dict["schema"] = SCHEMA_VERSION + "-quadratic_program"
    json.dump(qp_dict, f, indent=2)
    f.close()


def generate_and_save_reduced_quadratic_program_with_qpfs(redundancy_matrix, relevance_vector, num_active_features, num_frozen_features=0, alpha=None):

    reduced_redundancy_matrix, reduced_relevance_vector, constant_contribution, deselected_features, active_features, frozen_features = generate_reduced_quadratic_program_with_qpfs(redundancy_matrix, relevance_vector, num_active_features, num_frozen_features=0, alpha=None)
    save_quadratic_program(reduced_redundancy_matrix, reduced_relevance_vector, "reduced_quadratic_program.json", constant_term=constant_contribution)
    save_list(deselected_features, "deselected_features.json")
    save_list(active_features, "active_features.json")
    save_list(frozen_features, "frozen_features.json")


# np.random.seed(123)
# rand_mat = np.random.rand(6,6)
# test_Q = rand_mat @ np.transpose(rand_mat)
# test_f = np.random.rand(6,1)

# generate_and_save_reduced_quadratic_program_with_qpfs(test_Q, test_f, 3, num_frozen_features=2)
