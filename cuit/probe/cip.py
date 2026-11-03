"""CF Information Probing: text and SID-prefix probes, A_i, R_{i,l} and rho."""
import os

import numpy as np
import torch

from cuit.probe import probes as P
from cuit.probe.bank import ProbeBank


def fold_splits(y, pcfg):
    return [torch.as_tensor(P.make_folds(y.cpu().numpy(), pcfg.folds, pcfg.fold_seed + s), device=y.device)
            for s in range(pcfg.n_splits)]


def fit_probes(x, y, splits, cfg, l, log, tag):
    items = torch.arange(len(y), device=x.device)
    sets = [P.fit_crossfit_select(x, items, y, splits[0], cfg.cf.M, cfg.probe, cfg.seed + l, log, f"{tag} g{l}")]
    lam = sets[0].info["lambda_U"]
    for s in range(1, len(splits)):
        sets.append(P.fit_crossfit(x, items, y, splits[s], cfg.cf.M, cfg.probe, cfg.seed + 1000 * s + l, lam,
                                   lambda *a: None, f"{tag} s{s} g{l}"))
    log(f"  [{tag} g{l}] {len(sets)} splits, held-out nll {[round(g.info['heldout_nll'], 4) for g in sets]}")
    return sets


def text_probes(z, y, splits, cfg, log, cache):
    if os.path.exists(cache):
        return [P.CrossFitProbe.from_state(s, z.device) for s in torch.load(cache)]
    sets = fit_probes(z, y, splits, cfg, 0, log, "text")
    torch.save([g.state() for g in sets], cache)
    return sets


def prefix_vectors(codes, lengths, centroids, scale):
    L, N = centroids.shape[0], codes.shape[0]
    acc = torch.zeros(N, centroids.shape[2], device=centroids.device)
    out = []
    for l in range(L):
        has = lengths > l
        e = centroids[l][codes[:, l].clamp(max=centroids.shape[1] - 1)]
        acc = acc + torch.where(has[:, None], scale[l] * e, torch.zeros_like(e))
        out.append(acc)
    return torch.stack(out)


def kl_rows(logp, logq):
    return (logp.exp() * (logp - logq)).sum(-1)


def rho(A, R, A0):
    return (R / A.clamp_min(A0)).clamp(max=1.0)


@torch.no_grad()
def make_bank(z, depth_sets):
    items = torch.arange(len(depth_sets[0][0].folds), device=z.device)
    lpi = torch.stack([g.log_pi(items) for g in depth_sets[0]])
    log_pi = torch.logsumexp(lpi, 0) - np.log(len(depth_sets[0]))
    bank = ProbeBank(depth_sets, log_pi, z.device)
    logp0 = bank.log_q(0, z, items)
    return bank, logp0, kl_rows(logp0, log_pi)
