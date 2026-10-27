import numpy as np
import pandas as pd
import torch


def _tokens(value, sep=", "):
    if pd.isna(value):
        return []
    return sorted({t.strip() for t in str(value).split(sep) if t.strip()})


def build_features(users, items, user_ids, item_ids, use_metadata=True):
    vocab = {"<pad>": 0}

    def fid(name):
        if name not in vocab:
            vocab[name] = len(vocab)
        return vocab[name]

    users = users.set_index("user_id").reindex(user_ids)
    items = items.set_index("item_id").reindex(item_ids)

    user_rows = []
    for uid, row in enumerate(users.itertuples()):
        feats = [(fid(f"user={uid}"), 1.0)]
        if use_metadata:
            for col in ["age", "income", "sex", "kids_flg"]:
                value = getattr(row, col)
                feats.append((fid(f"{col}={'unknown' if pd.isna(value) else value}"), 1.0))
        user_rows.append(feats)

    item_rows = []
    for iid, row in enumerate(items.itertuples()):
        feats = [(fid(f"item={iid}"), 1.0)]
        if use_metadata:
            for col in ["genres", "countries"]:
                tokens = _tokens(getattr(row, col)) or ["unknown"]
                feats += [(fid(f"{col}={t}"), 1 / np.sqrt(len(tokens))) for t in tokens]
            year = row.release_year
            feats.append((fid(f"decade={'unknown' if pd.isna(year) else int(year) // 10 * 10}"), 1.0))
            rating = row.age_rating
            feats.append((fid(f"age_rating={'unknown' if pd.isna(rating) else int(rating)}"), 1.0))
        item_rows.append(feats)

    def pack(rows):
        width = max(len(r) for r in rows)
        idx = np.zeros((len(rows), width), dtype=np.int64)
        val = np.zeros((len(rows), width), dtype=np.float32)
        for n, feats in enumerate(rows):
            for m, (j, x) in enumerate(feats):
                idx[n, m], val[n, m] = j, x
        return idx, val

    return pack(user_rows), pack(item_rows), len(vocab)


class FactorizationMachine:
    def __init__(self, factors=32, reg=1e-4, lr=0.01, epochs=10, batch_size=4096, seed=42):
        self.factors = factors
        self.reg = reg
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

    def _side(self, idx, val):
        v = self.V[idx] * val[..., None]
        s = v.sum(1)
        inner = (self.w[idx] * val).sum(1) + 0.5 * ((s ** 2).sum(1) - (v ** 2).sum((1, 2)))
        return s, inner, v

    def _pair_score(self, users, items):
        su, _, vu = self._side(self.user_idx[users], self.user_val[users])
        si, inner_i, vi = self._side(self.item_idx[items], self.item_val[items])
        return inner_i + (su * si).sum(1), vu, vi

    def fit(self, seen, positives, features, callback=None):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        (u_idx, u_val), (i_idx, i_val), n_features = features
        self.user_idx, self.user_val = torch.from_numpy(u_idx), torch.from_numpy(u_val)
        self.item_idx, self.item_val = torch.from_numpy(i_idx), torch.from_numpy(i_val)

        self.V = (0.01 * torch.randn(n_features, self.factors)).requires_grad_()
        self.w = torch.zeros(n_features, requires_grad=True)
        with torch.no_grad():
            self.V[0] = 0  # пустой признак
        optimizer = torch.optim.Adam([self.V, self.w], lr=self.lr)

        n_items = seen.shape[1]
        pos_users, pos_items = positives.nonzero()
        seen_rows, seen_cols = seen.nonzero()
        seen_keys = np.sort(seen_rows.astype(np.int64) * n_items + seen_cols)

        self.history = []
        for epoch in range(1, self.epochs + 1):
            neg_items = rng.integers(0, n_items, len(pos_users))
            bad = np.isin(pos_users.astype(np.int64) * n_items + neg_items, seen_keys)
            while bad.any():
                neg_items[bad] = rng.integers(0, n_items, bad.sum())
                bad = np.isin(pos_users.astype(np.int64) * n_items + neg_items, seen_keys)

            order = rng.permutation(len(pos_users))
            total = 0.0
            for start in range(0, len(order), self.batch_size):
                batch = order[start:start + self.batch_size]
                u = torch.from_numpy(pos_users[batch])
                pos, vu, vp = self._pair_score(u, torch.from_numpy(pos_items[batch]))
                neg, _, vn = self._pair_score(u, torch.from_numpy(neg_items[batch]))
                bpr = -torch.nn.functional.logsigmoid(pos - neg).mean()
                l2 = self.reg * ((vu ** 2).sum() + (vp ** 2).sum() + (vn ** 2).sum()) / len(batch)
                loss = bpr + l2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += bpr.item() * len(batch)
            self.history.append(total / len(order))
            self._cache_items()
            if callback is not None:
                callback(self, epoch)
        return self

    def _cache_items(self):
        with torch.no_grad():
            all_items = torch.arange(self.item_idx.shape[0])
            s, inner, _ = self._side(self.item_idx[all_items], self.item_val[all_items])
            self._item_vectors, self._item_bias = s, inner

    def score(self, users):
        with torch.no_grad():
            users = torch.tensor(np.array(users, dtype=np.int64))
            su, _, _ = self._side(self.user_idx[users], self.user_val[users])
            return (su @ self._item_vectors.T + self._item_bias).numpy()
