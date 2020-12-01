import numpy as np
from cvxopt import matrix, solvers
from scipy.stats import pearsonr
from scipy.optimize import minimize, LinearConstraint
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

def construct_pearson_corr_relevance_vector(feature_matrix, label_vector):
    """Constructs the Pearson correlation relevance vector of a feature matrix and vector of discrete labels.
        This follows the construction of F on pg. 1495 of 'Rodriguez-Lujan, Irene, et al. "Quadratic programming 
        feature selection." Journal of Machine Learning Research (2010).'

    Args:
        feature_matrix (np.ndarray): 2D array with features along the columns and data points along the rows.
        label_vector (np.ndarray of ints): 1D array with each data point assigned to a column.

    Returns:
        relevancy_vector (np.ndarray): 1D array assigning a Pearson correlation relevancy score to each feature.
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

    relevancy_vector = np.zeros(num_of_features)
    for i in range(num_of_features):
        for j in range(num_of_classes):
            relevancy_vector[i] += np.abs(pearsonr(feature_matrix[:, i], binary_label_matrix[:, j])[0]) * class_probs[j]
    
    return relevancy_vector


def construct_mutual_info_relevance_vector(feature_matrix, label_vector, seed=None):
    """Constructs the mutual information relevance vector of a feature matrix and vector of discrete labels.
        See https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html

    Args:
        feature_matrix (np.ndarray): 2D array with features along the columns and data points along the rows.
        label_vector (np.ndarray of ints): 1D array with each data point assigned to a column.

    Returns:
        relevancy_vector (np.ndarray): 1D array assigning a mutual information relevancy score to each feature.
    """

    if seed is not None:
        np.random.seed(seed=seed)

    relevancy_vector =  mutual_info_classif(feature_matrix, label_vector)

    return relevancy_vector


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
    
    return redundancy_matrix


def _weight_features_with_quadratic_programming(redundancy_matrix, relevancy_vector, alpha):
    """Assigns an importance weight to each feature according to the output of a quadratic program following the methodology of
    'Rodriguez-Lujan, Irene, et al. "Quadratic programming feature selection." Journal of Machine Learning Research (2010).'

    Args:
        redundancy_matrix (np.ndarray): 2D square array assigning a correlation score to each pair of features.
        relevancy_vector (np.ndarray): 1D array assigning a relevancy score to each feature.
        alpha (float): parameter between 0 and 1 which weights the importance of relevancy (towards 0) vs redundancy (towards 1).

    Returns:
        weight_vector (np.ndarray): 1D array assigning an importance weight to each feature.
    """     

    num_of_features = len(relevancy_vector)
    P = matrix(redundancy_matrix*(1.0-alpha))
    q = matrix(-relevancy_vector*alpha)
    G = matrix(-np.eye(num_of_features))
    h = matrix(np.zeros(num_of_features))
    A = matrix(np.ones((1, num_of_features)))
    b = matrix(np.ones(1))
    opt_x = solvers.qp(P, q, G, h, A, b)['x']
    weight_vector = np.array(np.squeeze(opt_x))

    return weight_vector

def quadratic_programming_feature_selection(redundancy_matrix, relevancy_vector, num_of_chosen_features, alpha=None):
    """Selects a subset of features based on a quadratic program following the methodology of
    'Rodriguez-Lujan, Irene, et al. "Quadratic programming feature selection." Journal of Machine Learning Research (2010).'

    Args:
        redundancy_matrix (np.ndarray): 2D square array assigning a correlation score to each pair of features.
        relevancy_vector (np.ndarray): 1D array assigning a relevancy score to each feature.
        num_of_chosen_features (int): number of features to be selected
        alpha (float): parameter between 0 and 1 which weights the importance of relevancy (towards 0) vs redundancy (towards 1).

    Returns:
        chosen_ones (set): set of integers indexing the selected features.
        weight_vector (np.ndarray): 1D array assigning an importance weight to each of the original features.
    """

    # Set alpha as default if input is None
    if alpha is None:
        bar_q = np.mean(redundancy_matrix)
        bar_f = np.mean(relevancy_vector)
        alpha = bar_q / (bar_q + bar_f)

    feature_weights = _weight_features_with_quadratic_programming(redundancy_matrix, relevancy_vector, alpha)
    ranking = np.argsort(feature_weights)[::-1]
    chosen_ones = set(ranking[:num_of_chosen_features])

    return chosen_ones, feature_weights

def greedy_mrmr_feature_selection(redundancy_matrix, relevancy_vector, num_of_chosen_features, seed=None):
    """Selects a subset of features based on a the Minimum Redundancy Maximum Relevance method found here:
    https://en.wikipedia.org/wiki/Feature_selection#Minimum-redundancy-maximum-relevance_(mRMR)_feature_selection

    Args:
        redundancy_matrix (np.ndarray): 2D square array assigning a correlation score to each pair of features.
        relevancy_vector (np.ndarray): 1D array assigning a relevancy score to each feature.
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

    num_of_features = len(relevancy_vector)
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
            new_relevance = avg_relevance * len(chosen_ones) + relevancy_vector[i]
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
