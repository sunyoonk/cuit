"""Mini-batch k-means with k-means++ initialization."""
import torch


def sq_dist(x, c):
    return (x * x).sum(1, keepdim=True) - 2 * x @ c.T + (c * c).sum(1)[None]


def kmeans_pp(x, K, gen):
    c = torch.empty(K, x.shape[1], device=x.device, dtype=x.dtype)
    c[0] = x[torch.randint(len(x), (1,), generator=gen, device=x.device)]
    d = sq_dist(x, c[:1]).squeeze(1).clamp_min(0)
    for k in range(1, K):
        c[k] = x[torch.multinomial(d, 1, generator=gen)]
        d = torch.minimum(d, sq_dist(x, c[k:k + 1]).squeeze(1).clamp_min(0))
    return c


def _stream(n, batch_size, gen, device):
    while True:
        perm = torch.randperm(n, generator=gen, device=device)
        for s in range(0, n - batch_size + 1, batch_size):
            yield perm[s:s + batch_size]


def fit(x, K, kcfg, seed):
    gen = torch.Generator(device=x.device).manual_seed(seed)
    bs = min(kcfg.batch_size, len(x))
    stream = _stream(len(x), bs, gen, x.device)
    buf = []
    while sum(len(b) for b in buf) < min(kcfg.init_buffer, len(x)):
        idx = next(stream)
        buf.append(idx[: min(kcfg.init_buffer, len(x)) - sum(len(b) for b in buf)])
    c = kmeans_pp(x[torch.cat(buf)], K, gen)
    counts = torch.zeros(K, device=x.device)
    for _ in range(kcfg.steps):
        b = x[next(stream)]
        a = sq_dist(b, c).argmin(1)
        n = torch.bincount(a, minlength=K).float()
        counts += n
        m = n > 0
        means = (torch.nn.functional.one_hot(a, K).to(b.dtype).T @ b)[m] / n[m, None]
        c[m] = c[m] - (n[m] / counts[m])[:, None] * (c[m] - means)
    return c
