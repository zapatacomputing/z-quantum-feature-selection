import numpy as np
from cvxopt import matrix, solvers
from scipy.stats import pearsonr
from scipy.optimize import minimize, LinearConstraint
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

solvers.options['show_progress'] = False

def construct_pearson_corr_relevance_vector(feature_matrix, label_vector):
    """Constructs the Pearson correlation relevance vector of a feature matrix and vector of discrete labels.
        This follows the construction of F on pg. 1495 of 'Rodriguez-Lujan, Irene, et al. "Quadratic programming 
        feature selection." Journal of Machine Learning Research (2010).'

    Args:
        feature_matrix (np.ndarray): 2D array with features along the columns and data points along the rows.
        label_vector (np.ndarray of ints): 1D array with each data point assigned to a column.

    Returns:
        relevance_vector (np.ndarray): 1D array assigning a Pearson correlation relevance score to each feature.
    """

    # TODO: check if input label vector has integer values
    num_of_samples, num_of_features = feature_matrix.shape

    classes = list(set(label_vector))
    num_of_classes = len(classes)
    class_probs = np.zeros(num_of_classes)
    binary_label_matrix = np.zeros((num_of_samples, num_of_classes))
    for j in range(num_of_classes):
        binary_label_matrix[:, j] = np.where(label_vector==classes[j], 1.0, 0.0)
        class_probs[j] = np.sum(binary_label_matrix[:, j]) / num_of_samples

    relevance_vector = np.zeros(num_of_features)
    for i in range(num_of_features):
        for j in range(num_of_classes):
            relevance_vector[i] += np.abs(pearsonr(feature_matrix[:, i], binary_label_matrix[:, j])[0]) * class_probs[j]
    
    return relevance_vector


def construct_mutual_info_relevance_vector(feature_matrix, label_vector, seed=None):
    """Constructs the mutual information relevance vector of a feature matrix and vector of discrete labels.
        See https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html

    Args:
        feature_matrix (np.ndarray): 2D array with features along the columns and data points along the rows.
        label_vector (np.ndarray of ints): 1D array with each data point assigned to a column.

    Returns:
        relevance_vector (np.ndarray): 1D array assigning a mutual information relevance score to each feature.
    """

    if seed is not None:
        np.random.seed(seed=seed)

    relevance_vector =  mutual_info_classif(feature_matrix, label_vector)

    return relevance_vector


def construct_pearson_corr_redundancy_matrix(feature_matrix):
    """Constructs the Pearson correlation redundancy matrix of a feature matrix.
        This follows the construction of rho_ij in eq. 3 of 'Rodriguez-Lujan, Irene, et al. "Quadratic programming 
        feature selection." Journal of Machine Learning Research (2010).'

    Args:
        feature_matrix (np.ndarray): 2D array with features along the columns and data points along the rows.

    Returns:
        redundancy_matrix (np.ndarray): 2D square array assigning a Pearson correlation score to each pair of features.
    """        

    redundancy_matrix = np.abs(np.corrcoef(feature_matrix.T))

    return redundancy_matrix


def construct_mutual_info_redundancy_matrix(feature_matrix, seed=None):
    """Constructs the mutual information redundancy matrix of a feature matrix.
    See https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html

    Args:
        feature_matrix (np.ndarray): 2D array with features along the columns and data points along the rows.
        seed (int): random seed.

    Returns:
        redundancy_matrix (np.ndarray): 2D square array assigning a mutual information score to each pair of features.
    """      
    if seed is not None:
        np.random.seed(seed=seed)
    num_of_features = len(feature_matrix[0])
    
    # Compute mutual information between data points
    list_of_mutual_info_vectors = [mutual_info_regression(feature_matrix, feature_matrix[:, i]) for i in range(num_of_features)]
    
    redundancy_matrix =  np.vstack(list_of_mutual_info_vectors)
    redundancy_matrix = (redundancy_matrix + redundancy_matrix.T) / 2.0

    return redundancy_matrix


def _weight_features_with_quadratic_programming(redundancy_matrix, relevance_vector, alpha):
    """Assigns an importance weight to each feature according to the output of a quadratic program following the methodology of
    'Rodriguez-Lujan, Irene, et al. "Quadratic programming feature selection." Journal of Machine Learning Research (2010).'

    Args:
        redundancy_matrix (np.ndarray): 2D square array assigning a correlation score to each pair of features.
        relevance_vector (np.ndarray): 1D array assigning a relevance score to each feature.
        alpha (float): parameter between 0 and 1 which weights the importance of relevance (towards 1) vs redundancy (towards 0).

    Returns:
        weight_vector (np.ndarray): 1D array assigning an importance weight to each feature.
    """     

    num_of_features = len(relevance_vector)
    P = matrix(redundancy_matrix*(1.0-alpha))
    q = matrix(-relevance_vector*alpha)
    G = matrix(-np.eye(num_of_features))
    h = matrix(np.zeros(num_of_features))
    A = matrix(np.ones((1, num_of_features)))
    b = matrix(np.ones(1))
    opt_x = solvers.qp(P, q, G, h, A, b)['x']
    weight_vector = np.array(np.squeeze(opt_x))

    return weight_vector

def quadratic_programming_feature_selection(redundancy_matrix, relevance_vector, num_of_chosen_features, alpha=None):
    """Selects a subset of features based on a quadratic program following the methodology of
    'Rodriguez-Lujan, Irene, et al. "Quadratic programming feature selection." Journal of Machine Learning Research (2010).'

    Args:
        redundancy_matrix (np.ndarray): 2D square array assigning a correlation score to each pair of features.
        relevance_vector (np.ndarray): 1D array assigning a relevance score to each feature.
        num_of_chosen_features (int): number of features to be selected
        alpha (float): parameter between 0 and 1 which weights the importance of relevance (towards 1) vs redundancy (towards 0).

    Returns:
        chosen_ones (list): list of integers indexing the selected features.
        weight_vector (np.ndarray): 1D array assigning an importance weight to each of the original features.
    """
    # Ensure that the redundancy matrix is symmetric
    redundancy_matrix = (redundancy_matrix + redundancy_matrix.T) / 2.0

    # Set alpha as default if input is None
    if alpha is None:
        bar_q = np.mean(redundancy_matrix)
        bar_f = np.mean(relevance_vector)
        alpha = bar_q / (bar_q + bar_f)

    feature_weights = _weight_features_with_quadratic_programming(redundancy_matrix, relevance_vector, alpha)
    ranking = np.argsort(feature_weights)[::-1]
    # chosen_ones = set(ranking[:num_of_chosen_features])
    # Keeping as list so that the assignment of indices to features is preserved in output
    chosen_ones = ranking[:num_of_chosen_features]

    return chosen_ones, feature_weights

def greedy_mrmr_feature_selection(redundancy_matrix, relevance_vector, num_of_chosen_features, seed=None):
    """Selects a subset of features based on a the Minimum Redundancy Maximum Relevance method found here:
    https://en.wikipedia.org/wiki/Feature_selection#Minimum-redundancy-maximum-relevance_(mRMR)_feature_selection

    Args:
        redundancy_matrix (np.ndarray): 2D square array assigning a correlation score to each pair of features.
        relevance_vector (np.ndarray): 1D array assigning a relevance score to each feature.
        num_of_chosen_features (int): number of features to be selected
        seed (int): random seed.

    Returns:
        chosen_ones (set): set of integers indexing the selected features.
        mrmr (float): final mrmr score of selected subset of features.
        avg_relevance: average relevance score of selected subset of features.
        avg_redundancy: average redundancy score of selected subset of features.
    """  

    if seed is not None:
        np.random.seed(seed=seed)

    # Ensure that the redundancy matrix is symmetric
    redundancy_matrix = (redundancy_matrix + redundancy_matrix.T) / 2.0

    num_of_features = len(relevance_vector)
    chosen_ones = set()
    avg_relevance = 0.0
    avg_redundancy = 0.0
    mrmr = avg_relevance - avg_redundancy

    # Add feature indexes until k have been chosen
    while len(chosen_ones) < num_of_chosen_features:
        best_mrmr = None
        best_feature_idx = None

        # Find the best feature from the remaining set according to mrmr (randomizing order in
        # case two features have the same mrmr)
        for i in np.random.permutation(range(num_of_features)):

            # Skip if i has already been chosen
            if i in chosen_ones:
                continue

            # Compute mrmr when feature is included
            new_relevance = avg_relevance * len(chosen_ones) + relevance_vector[i]
            new_avg_relevance = new_relevance / (len(chosen_ones) + 1)
            new_redundancy = avg_redundancy * len(chosen_ones)**2 + 2 * np.sum([redundancy_matrix[i, k] for k in chosen_ones]) + redundancy_matrix[i, i]
            new_avg_redundancy = new_redundancy / (len(chosen_ones) + 1)**2
            new_mrmr = new_avg_relevance - new_avg_redundancy

            # Update best_mrmr and feature index if better is found
            if best_mrmr is None or new_mrmr > best_mrmr:
                best_mrmr = new_mrmr
                best_feature_idx = i
                best_avg_relevance = new_avg_relevance
                best_avg_redundacy = new_avg_redundancy

        # Add new feature to chosen ones and update optimization quantities
        chosen_ones.add(best_feature_idx)
        mrmr = best_mrmr
        avg_relevance = best_avg_relevance
        avg_redundancy = best_avg_redundacy

    return chosen_ones, mrmr, avg_relevance, avg_redundancy


def generate_reduced_quadratic_program_with_qpfs(redundancy_matrix, relevance_vector, num_active_features, constant_contribution=np.zeros(1), num_frozen_features=0, alpha=None, frozen_vector_strategy="QUBO"):
    """Generates a reduced instance of the quadratic programming feature selection problem. Splits features into frozen, active, and deselected 
    corresponding respectively to those automatically chosen, those to be used in the reduced instance, and those to be automatically deselected.
    The reduced instances are constructed by evaluating certain entries in the optimization argument x to yield an effective constrained instance:
    Letting 
    
    x = x_d \oplus x_a \oplus x_f, 
    
    where 
    x_d is the all-zero vector on the "deslected features"
    x_a is the remaining optimization variable on the "active features"
    x_f is the vector of weight assignments for the "frozen features" (the weights vary according to the frozen_vector_strategy)

    The entries corresponding to these vectors are determined by the chosen number of corresponding features, the weights of the full QPFS weight
    assignment, and the frozen_vector_strategy (see below). These assignments result in the following reduced quadratic program:
    
    0.5 * (x_d \oplus x_a \oplus x_f) Q (x_d \oplus x_a \oplus x_f) + f (x_d \oplus x_a \oplus x_f)
    = 0.5 * x_a Q' x_a + f' x_a + c

    where
    Q' is the submatrix of Q of active features
    f' is the subvector of f of active features plus (x_f * Q) (arising from the cross terms in the first term)
    c  is the constant contribution formed by 0.5 (x_f Q x_f) + (f x_f)

    There are several optional strategies for constructing this reduces quadratic program. They vary according to whether the optimal value
    is consistent with the reduced QUBO problem, the reduced QPFS problem, or a hybrid of the two. These are as follows:
    "QUBO": frozen vector is an array of all ones, corresponding to the QUBO automatically selecting those features.
    "QPFS": frozen vector is the subvector of feature weights from the full-instance QPFS weight assignments.
    "hybrid": frozen vector is the uniform distribution with total weight given by the sum of the frozen feature weights from QPFS.

    Args:
        redundancy_matrix (np.ndarray): 2D square array assigning a correlation score to each pair of features.
        relevance_vector (np.ndarray): 1D array assigning a relevance score to each feature.
        num_active_features(int): number of features of reduced problem
        num_frozen_features(int): number of features to be automatically selected
        alpha (float): parameter between 0 and 1 which weights the importance of relevance (towards 1) vs redundancy (towards 0).
        frozen_vector_strategy (string): string indicating the strategy for constructing the frozen vector (e.g. "QUBO", "QPFS", "hybrid")

    Returns:
        reduced_redundancy_matrix (np.ndarray): 2D square array assigning a correlation score to each pair of active features.
        reduced_relevance_vector (np.ndarray): 1D array assigning a relevance score to the active features.
        constant_contribution (np.array): single number representing the constant contribution to the QP score from frozen features
        deselected_features (list): list of features not to be selected
        active_features (list): list of features to include in reduced problem
        frozen_features (list): of features automatically selected in reduced problem
    """ 

    num_features = relevance_vector.size

    # Ensure that the redundancy matrix is symmetric
    redundancy_matrix = (redundancy_matrix + redundancy_matrix.T) / 2.0

    # Set alpha as default if input is None
    if alpha is None:
        bar_q = np.mean(redundancy_matrix)
        bar_f = np.mean(relevance_vector)
        alpha = bar_q / (bar_q + bar_f)

    _, feature_weights = quadratic_programming_feature_selection(redundancy_matrix, 
                                                                relevance_vector, 
                                                                num_features,
                                                                alpha=alpha)

    # print("redundancy_matrix", redundancy_matrix)
    # print("relevance_vector", relevance_vector)
    # print("feature_weights", feature_weights)
    # Check to make sure that num_frozen + num_selected is less than total
    if num_active_features + num_frozen_features > num_features:
        raise ValueError("Number of active features plus number of frozen features must be less than number of features in relevance")
    

    # For each rank (0 is lowest), list entry is the feature index
    # sorted_features = np.argsort(feature_weights)[::-1]
    sorted_features = np.argsort(feature_weights)
    # print("sorted_features", sorted_features)

    # Get lists of frozen, active, and deselected features
    num_deselected_features = num_features - (num_active_features + num_frozen_features)
    deselected_features = sorted_features[:num_deselected_features]
    active_features = sorted_features[num_deselected_features:num_deselected_features+num_active_features]
    frozen_features = sorted_features[num_deselected_features+num_active_features:]
    # print(deselected_features, active_features, frozen_features)

    # Get the effective redundancy matrix and relevance vector
    effective_redundancy_matrix = (1 - alpha) * redundancy_matrix
    effective_relevance_vector = - alpha * relevance_vector

    ### Compute reduced redundancy matrix by simply extracting the submatrix of active features
    reduced_redundancy_matrix = effective_redundancy_matrix[np.ix_(active_features, active_features)] 
    # print("reduced_redundancy_matrix", reduced_redundancy_matrix)

    ### Generate frozen weights vector according to strategy
    if frozen_vector_strategy == "QUBO":
        frozen_weights_vector = np.ones(num_frozen_features)

    elif frozen_vector_strategy == "QPFS":
        frozen_weights_vector = feature_weights[frozen_features]

    elif frozen_vector_strategy == "hybrid":
        weight_of_frozen_features = np.sum(feature_weights[frozen_features])
        frozen_weights_vector = weight_of_frozen_features * np.ones(num_frozen_features) / num_frozen_features

    else:
        raise ValueError(f"Frozen vector strategy {frozen_vector_strategy} not currently supported. Please select consult doc strings for valid options.")

    # print("frozen_weights_vector", frozen_weights_vector)
    
    ### Compute reduced relevance vector
    frozen_active_redundancy_matrix = effective_redundancy_matrix[np.ix_(frozen_features, active_features)] 
    # print("frozen_active_redundancy_matrix", frozen_active_redundancy_matrix)
    frozen_redundancy_contribution_vector = frozen_weights_vector @ frozen_active_redundancy_matrix
    # print("frozen_redundancy_contribution_vector", frozen_redundancy_contribution_vector)
    reduced_relevance_vector = frozen_redundancy_contribution_vector + effective_relevance_vector[active_features] 
    # print("reduced_relevance_vector", reduced_relevance_vector)

    ### TODO: Alternatively, add constant term to diagonal of redundandy matrix?
    frozen_redundancy_matrix = effective_redundancy_matrix[np.ix_(frozen_features, frozen_features)] 
    # print("frozen_redundancy_matrix", frozen_redundancy_matrix)
    frozen_relevance_vector = effective_relevance_vector[frozen_features]
    # print("frozen_relevance_vector", frozen_relevance_vector)
    constant_from_frozen_redundancy = np.dot(frozen_weights_vector, frozen_redundancy_matrix @ frozen_weights_vector) * 0.5
    # print("constant_from_frozen_redundancy", constant_from_frozen_redundancy)
    constant_from_frozen_relevancy = frozen_weights_vector @ frozen_relevance_vector
    # print("constant_from_frozen_relevancy", constant_from_frozen_relevancy)
    constant_contribution = constant_from_frozen_redundancy + constant_from_frozen_relevancy
    # print("constant_contribution", constant_contribution)

    return reduced_redundancy_matrix, reduced_relevance_vector, constant_contribution, deselected_features, active_features, frozen_features