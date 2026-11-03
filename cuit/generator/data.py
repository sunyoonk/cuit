"""Training windows and evaluation batches for the generator."""
import numpy as np
import torch


class TrainWindows(torch.utils.data.IterableDataset):
    def __init__(self, train_seqs, path, sid, g, seed):
        self.seqs = [np.asarray(s) for s in train_seqs]
        self.path, self.sid = path.numpy(), sid.numpy()
        self.users, self.windows, self.seed = g.users_per_batch, g.windows_per_batch, seed
        self.max_items = g.max_tokens // sid.shape[1]

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([self.seed, 0 if info is None else info.id])
        while True:
            order = rng.permutation(len(self.seqs))
            for s in range(0, len(order) - self.users + 1, self.users):
                yield self._batch([self.seqs[u] for u in order[s:s + self.users]], rng)

    def _batch(self, seqs, rng):
        n = np.array([len(s) for s in seqs])
        per_user = n * (n - 1) // 2
        total = int(per_user.sum())
        pick = np.unique(rng.integers(0, total, self.windows))
        u = np.searchsorted(np.cumsum(per_user), pick, side="right")
        j = pick - np.r_[0, np.cumsum(per_user)][u]
        tri = np.arange(1, n.max() + 1) * np.arange(2, n.max() + 2) // 2
        b = np.searchsorted(tri, j, side="right") + 2
        a = j - (b - 2) * (b - 1) // 2
        a = np.maximum(a, b - self.max_items)
        T = int((b - a - 1).max())
        L = self.path.shape[1]
        hist = np.full((len(pick), T, L), -1, np.int64)
        tgt = np.empty((len(pick), L), np.int64)
        for r, (uu, aa, bb) in enumerate(zip(u, a, b)):
            items = seqs[uu][aa:bb]
            hist[r, :len(items) - 1] = self.path[items[:-1]]
            tgt[r] = self.sid[items[-1]]
        return torch.from_numpy(hist), torch.from_numpy(tgt)


def eval_batches(seqs, path, batch_size):
    path = path.numpy()
    L = path.shape[1]
    for s in range(0, len(seqs), batch_size):
        chunk = seqs[s:s + batch_size]
        T = max(len(x) - 1 for x in chunk)
        hist = np.full((len(chunk), T, L), -1, np.int64)
        for r, x in enumerate(chunk):
            hist[r, :len(x) - 1] = path[x[:-1]]
        yield np.arange(s, s + len(chunk)), torch.from_numpy(hist), torch.as_tensor([x[-1] for x in chunk])
