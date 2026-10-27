import numpy as np
from scipy.sparse import csr_matrix


class ItemKNN:
    def __init__(self, k=200, shrinkage=50):
        self.k = k                  
        self.shrinkage = shrinkage  

    def fit(self, X):
        self.X = X.astype(np.float32).tocsr()
        co = (self.X.T @ self.X).toarray()
        norms = np.sqrt(np.diag(co))  

        sim = co / (np.outer(norms, norms) + self.shrinkage + 1e-9)
        np.fill_diagonal(sim, 0)  

        k = min(self.k, sim.shape[0] - 1)
        neighbours = np.argpartition(-sim, k, axis=1)[:, :k]
        rows = np.repeat(np.arange(sim.shape[0]), k)
        cols = neighbours.ravel()
        self.sim = csr_matrix((sim[rows, cols], (rows, cols)), shape=sim.shape)
        self.sim.eliminate_zeros()
        return self

    def score(self, users):
        return (self.X[users] @ self.sim).toarray()

    def neighbours(self, item, n=5):
        row = self.sim.getrow(item)
        order = np.argsort(-row.data, kind="stable")[:n]
        return row.indices[order], row.data[order]
