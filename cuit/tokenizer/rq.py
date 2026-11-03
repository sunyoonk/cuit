"""RQ-Kmeans and full-path extension."""
import torch
import torch.nn.functional as F

from cuit.tokenizer import kmeans


def kmeans_seed(seed, l):
    return 1000 * seed + l


@torch.no_grad()
def rq_kmeans(z, K, L, kcfg, seed):
    r, cum = z.float(), z.float().norm(dim=1)
    codes, cents, scale = [], [], []
    for l in range(L):
        rb = F.normalize(r, dim=1)
        c = kmeans.fit(rb, K, kcfg, kmeans_seed(seed, l))
        t = kmeans.sq_dist(rb, c).argmin(1)
        scale.append(float(cum.mean()))
        r = rb - c[t]
        cum = cum * r.norm(dim=1)
        codes.append(t)
        cents.append(c)
    return torch.stack(codes, 1), torch.stack(cents), torch.tensor(scale)


@torch.no_grad()
def extend_paths(z, codes, lengths, cents):
    r = z.float()
    path = codes.clone()
    for l in range(cents.shape[0]):
        rb = F.normalize(r, dim=1)
        geo = kmeans.sq_dist(rb, cents[l]).argmin(1)
        t = torch.where(lengths > l, codes[:, l], geo)
        path[:, l] = t
        r = rb - cents[l][t]
    return path
