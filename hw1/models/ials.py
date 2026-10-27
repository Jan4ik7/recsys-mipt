import numpy as np


def solve_side(fixed, P, alpha, reg):
    factors = fixed.shape[1]
    YtY = fixed.T @ fixed + reg * np.eye(factors)  
    result = np.zeros((P.shape[0], factors))
    for u in range(P.shape[0]):
        start, end = P.indptr[u], P.indptr[u + 1]
        if start == end:
            continue
        Y_u = fixed[P.indices[start:end]]
        p_u = P.data[start:end]
        A = YtY + alpha * Y_u.T @ Y_u
        b = (1 + alpha) * Y_u.T @ p_u
        result[u] = np.linalg.solve(A, b)
    return result


class IALS:
    def __init__(self, factors=32, alpha=40, reg=1000, iterations=10, seed=42):
        self.factors = factors
        self.alpha = alpha
        self.reg = reg
        self.iterations = iterations
        self.seed = seed

    def fit(self, P, callback=None):
        self.P = P.astype(np.float64).tocsr()
        P_items = self.P.T.tocsr()
        rng = np.random.default_rng(self.seed)
        self.item_factors = rng.normal(0, 0.01, (P.shape[1], self.factors))
        self.user_factors = np.zeros((P.shape[0], self.factors))
        self.history = []
        for iteration in range(1, self.iterations + 1):
            self.user_factors = solve_side(self.item_factors, self.P, self.alpha, self.reg)
            self.item_factors = solve_side(self.user_factors, P_items, self.alpha, self.reg)
            self.history.append(self.loss())
            if callback is not None:
                callback(self, iteration)
        return self

    def loss(self):
        X, Y, P = self.user_factors, self.item_factors, self.P
        rows = np.repeat(np.arange(P.shape[0]), np.diff(P.indptr))
        pred = np.einsum("ij,ij->i", X[rows], Y[P.indices])
        total = np.sum((X.T @ X) * (Y.T @ Y))
        total += np.sum((1 + self.alpha) * (P.data - pred) ** 2 - pred ** 2)
        return float(total + self.reg * (np.sum(X ** 2) + np.sum(Y ** 2)))

    def score(self, users):
        return self.user_factors[users] @ self.item_factors.T
