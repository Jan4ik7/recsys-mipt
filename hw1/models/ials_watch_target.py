import numpy as np

from .ials import IALS


def watch_target(watched_pct, floor=0.5):
    pct = np.asarray(watched_pct, dtype=float)
    pct = np.where(np.isnan(pct), np.nanmedian(pct), pct)  # пропуски заполняем медианой
    return floor + (1 - floor) * np.clip(pct / 50, 0, 1)


class IALSWatchTarget(IALS):
    def __init__(self, floor=0.5, **kwargs):
        super().__init__(**kwargs)
        self.floor = floor

    def fit(self, X, watched_pct, callback=None):
        assert len(watched_pct) == X.nnz
        P = X.astype(np.float64).tocsr(copy=True)
        P.data = watch_target(watched_pct, self.floor)
        return super().fit(P, callback)
