import numpy as np
from scipy.linalg import norm


def cosine_similarity(a, b):
    a = np.squeeze(a)
    b = np.squeeze(b)
    return np.dot(a, b) / (norm(a) * norm(b))