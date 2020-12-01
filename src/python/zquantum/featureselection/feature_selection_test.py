import pytest
import numpy as np
from feature_selection import (construct_mutual_info_relevance_vector,
                               construct_pearson_corr_relevance_vector,
                               construct_mutual_info_redundancy_matrix,
                               construct_pearson_corr_redundancy_matrix,
                               quadratic_programming_feature_selection,
                               greedy_mrmr_feature_selection)

# The dataset consists of 10 points around (0, 0, 0) and 10 points around (10, 10, 10)
# The feature vector for point (x, y, z) is (x, y, z, sin(x))
X = np.array([[-0.8,  -1.04,  0.53, -0.72],
            [ 1.13, -0.69, -1.06,  0.91],
            [ 2.01,  0.41, -0.53,  0.9 ],
            [-0.28,  0.04, -0.71, -0.28],
            [-0.86, -0.87, -0.39, -0.76],
            [ 1.1,   0.6, -0.03,  0.89],
            [ 0.78, -0.62,  0.92,  0.71],
            [-1.07, -0.53,  1.39, -0.88],
            [ 0.08,  0.37,  0.16,  0.08],
            [-0.95, -1.27, -0.05, -0.81],
            [ 9.2,   8.96, 10.53,  0.22],
            [11.13,  9.31,  8.94, -0.99],
            [12.01, 10.41,  9.47, -0.53],
            [ 9.72, 10.04,  9.29, -0.29],
            [ 9.14,  9.13,  9.61,  0.28],
            [11.1,  10.6,   9.97, -0.99],
            [10.78,  9.38, 10.92, -0.98],
            [ 8.93,  9.47, 11.39,  0.48],
            [10.08, 10.37, 10.16, -0.61],
            [ 9.05,  8.73,  9.95,  0.37]])

y = np.array([0]*10+[1]*10).astype(np.int32)

f_pc = np.array([0.97960675, 0.99216622, 0.98986846, 0.22647794])
Q_pc = np.array([[1.,         0.98749687, 0.95835005, 0.18655343],
                [0.98749687, 1.,         0.97879136, 0.21414268],
                [0.95835005, 0.97879136, 1.,         0.23262081],
                [0.18655343, 0.21414268, 0.23262081, 1.        ]])

f_mi = np.array([0.7187714,  0.7187714,  0.7187714,  0.22470394])
Q_mi = np.array([[1.71440632, 0.79166823, 0.63154918, 0.57452537],
                [0.79166823, 1.71440632, 0.6415095,  0.3495849 ],
                [0.63154918, 0.6415095,  1.71440632, 0.10748172],
                [0.57452537, 0.3495849,  0.10748172, 1.68523966]])

num_chosen_features = 3

qpfs_chosen_features_pc = set([0, 1, 2])
qpfs_feature_weights_pc = np.array([0.24065306, 0.25723705, 0.43544254, 0.06666735])

qpfs_chosen_features_mi = set([0, 1, 2])
qpfs_feature_weights_mi = np.array([0.3144958,  0.31306083, 0.3622878,  0.01015557])

mrmr_chosen_features_pc = set([0, 1, 3])
mrmr_mrmr_pc = 0.09092964032036388
mrmr_avg_relevance_pc = 0.7327503035940609
mrmr_avg_redundancy_pc = 0.641820663273697

mrmr_chosen_features_mi = set([1, 2, 3])
mrmr_mrmr_mi = -0.2582738095238094
mrmr_avg_relevance_mi = 0.5540822497362753
mrmr_avg_redundancy_mi = 0.8123560592600847

def test_construct_pearson_corr_relevance_vector():
    res = construct_pearson_corr_relevance_vector(X, y)
    assert res.shape == f_pc.shape
    assert np.isclose(res, f_pc).all()

def test_construct_mutual_info_relevance_vector():
    res = construct_mutual_info_relevance_vector(X, y, seed=42)
    assert res.shape == f_mi.shape
    assert np.isclose(res, f_mi).all()

def test_construct_pearson_corr_redundancy_matrix():
    res = construct_pearson_corr_redundancy_matrix(X)
    assert res.shape == Q_pc.shape
    assert np.isclose(res, Q_pc).all()

def test_construct_mutual_info_redundancy_matrix():
    res = construct_mutual_info_redundancy_matrix(X, seed=42)
    assert res.shape == Q_mi.shape
    assert np.isclose(res, Q_mi).all()

def test_quadratic_programming_feature_selection():
    chosen_ones_pc, feature_weights_pc = quadratic_programming_feature_selection(Q_pc, f_pc, num_chosen_features, alpha=None)
    assert chosen_ones_pc == qpfs_chosen_features_pc
    assert feature_weights_pc.shape == qpfs_feature_weights_pc.shape
    assert np.isclose(feature_weights_pc, qpfs_feature_weights_pc).all()

    chosen_ones_mi, feature_weights_mi = quadratic_programming_feature_selection(Q_mi, f_mi, num_chosen_features, alpha=None)
    assert chosen_ones_mi == qpfs_chosen_features_mi
    assert feature_weights_mi.shape == qpfs_feature_weights_mi.shape
    assert np.isclose(feature_weights_mi, qpfs_feature_weights_mi).all()

def test_greedy_mrmr_feature_selection():
    chosen_ones_pc, mrmr_pc, avg_relevance_pc, avg_redundancy_pc = greedy_mrmr_feature_selection(Q_pc, f_pc, num_chosen_features, seed=42)
    assert chosen_ones_pc == mrmr_chosen_features_pc
    assert np.isclose(mrmr_pc, mrmr_mrmr_pc).all()
    assert np.isclose(avg_relevance_pc, mrmr_avg_relevance_pc).all()
    assert np.isclose(avg_redundancy_pc, mrmr_avg_redundancy_pc).all()

    chosen_ones_mi, mrmr_mi, avg_relevance_mi, avg_redundancy_mi = greedy_mrmr_feature_selection(Q_mi, f_mi, num_chosen_features, seed=42)
    assert chosen_ones_mi == mrmr_chosen_features_mi
    assert np.isclose(mrmr_mi, mrmr_mrmr_mi).all()
    assert np.isclose(avg_relevance_mi, mrmr_avg_relevance_mi).all()
    assert np.isclose(avg_redundancy_mi, mrmr_avg_redundancy_mi).all()
