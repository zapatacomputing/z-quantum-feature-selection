import numpy as np
from zquantum.qubo import save_qubo
from zquantum.featureselection import prepare_qubo_for_feature_selection


def generate_qubo_for_feature_selection(x: np.ndarray, y: np.ndarray, alpha: float):
    qubo = prepare_qubo_for_feature_selection(x, y, alpha)
    save_qubo(qubo, "qubo.json")
