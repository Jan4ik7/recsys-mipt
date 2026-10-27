import numpy as np
from scipy.linalg import cho_factor, cho_solve


class EASE:
    def __init__(self, reg=1000):
        self.reg = reg

    def fit(self, X):
        self.X = X.astype(np.float32).tocsr()
        G = (X.T @ X).toarray().astype(np.float64)   
        G[np.diag_indices_from(G)] += self.reg
        P = cho_solve(cho_factor(G), np.eye(G.shape[0]))  
        B = -P / np.diag(P)[None, :]
        np.fill_diagonal(B, 0)
        self.B = B.astype(np.float32)
        return self

    def score(self, users):
        return np.asarray(self.X[users] @ self.B)
