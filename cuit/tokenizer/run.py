"""CUIT tokenizer.

    python -m cuit.tokenizer.run dataset=beauty seed=42
"""
import json
import os

import numpy as np
import torch

from cuit import config as C
from cuit.probe import cip
from cuit.probe import probes as P
from cuit.tokenizer import rq
from cuit.tokenizer.igq import igq_round


def check_sids(sid, lengths, K):
    L = sid.shape[1]
    assert ((sid == K) == (torch.arange(L, device=sid.device)[None] >= lengths[:, None])).all()
    assert len(torch.unique(sid, dim=0)) == len(sid), "SIDs are not unique"
    for d in range(1, L):
        short = {tuple(x) for x in sid[lengths == d, :d].tolist()}
        assert not any(tuple(x) in short for x in sid[lengths > d, :d].tolist()), "an SID is a prefix of another"


def sid_probes(z, y, splits, cfg, codes, lengths, cents, scale, log, tag, cache=None):
    if cache and os.path.exists(cache):
        log(f"  [{tag}] SID probes from {cache}")
        return [[P.CrossFitProbe.from_state(s, z.device) for s in g] for g in torch.load(cache)]
    pre = cip.prefix_vectors(codes, lengths, cents, scale.tolist())
    sets = [cip.fit_probes(pre[l], y, splits, cfg, l + 1, log, tag) for l in range(cents.shape[0])]
    if cache:
        torch.save([[g.state() for g in gs] for gs in sets], cache)
    return sets


def main():
    cfg = C.load_config()
    out = os.path.join(cfg.out_dir, "tokenizer")
    C.save_config(cfg, out, "tokenizer")
    log = C.Logger(os.path.join(out, "tokenizer.log"))
    C.seed_all(cfg.seed)
    dev = "cuda"
    t = cfg.tok
    z = torch.load(os.path.join(cfg.work_dir, "text_emb.pt")).float().to(dev)
    y = torch.as_tensor(np.load(os.path.join(cfg.work_dir, "cf", "y.npy")), device=dev)
    splits = cip.fold_splits(y, cfg.probe)
    log(f"[tok] N={len(z)} labeled={int((y >= 0).sum())} K={t.K} L_max={t.L_max} lam={t.lam} eps={t.eps} "
        f"delta={t.delta} J={t.rounds}")

    codes, cents, scale = rq.rq_kmeans(z, t.K, t.L_max, t.kmeans, cfg.seed)
    lengths = torch.full((len(z),), t.L_max, device=dev)
    torch.save({"codes": codes.cpu(), "centroids": cents.cpu(), "scale": scale}, os.path.join(out, "round0.pt"))
    log(f"[round 0] RQ-Kmeans: {len(torch.unique(codes, dim=0))} distinct of {len(z)}")
    g_text = cip.text_probes(z, y, splits, cfg, log, os.path.join(out, "text_probes.pt"))
    g_sid = sid_probes(z, y, splits, cfg, codes, lengths, cents, scale, log, "round0",
                       os.path.join(out, "round0_probes.pt"))
    bank, logp0, A = cip.make_bank(z, [g_text] + g_sid)
    A0 = float(A.median())
    log(f"[round 0] A: mean={float(A.mean()):.4f} A_0 (median)={A0:.4f} items with A <= delta: {int((A <= t.delta).sum())}")

    for r in range(1, t.rounds + 1):
        log(f"[round {r}] IGQ")
        res = igq_round(z, y, bank, logp0, A, A0, cfg, log)
        check_sids(res["codes"], res["lengths"], t.K)
        hist = torch.bincount(res["lengths"], minlength=t.L_max + 1)[1:].tolist()
        log(f"[round {r}] mean length {res['lengths'].float().mean():.4f}, lengths {hist}")
        torch.save({k: (v.cpu() if torch.is_tensor(v) else v) for k, v in res.items()}, os.path.join(out, f"round{r}.pt"))
        if r < t.rounds:
            g_sid = sid_probes(z, y, splits, cfg, res["codes"], res["lengths"], res["centroids"], res["scale"], log,
                               f"round{r}")
            bank, _, _ = cip.make_bank(z, [g_text] + g_sid)

    path = rq.extend_paths(z, res["codes"], res["lengths"], res["centroids"])
    on = torch.arange(t.L_max, device=dev)[None] < res["lengths"][:, None]
    assert torch.equal(path[on], res["codes"][on])
    torch.save({"sid": res["codes"].cpu(), "path": path.cpu(), "lengths": res["lengths"].cpu(), "K": t.K},
               os.path.join(out, "sid.pt"))
    json.dump({"mean_length": float(res["lengths"].float().mean()), "lengths": hist, "stats": res["stats"]},
              open(os.path.join(out, "summary.json"), "w"), indent=1)
    log(f"[tok] SIDs -> {out}/sid.pt")


if __name__ == "__main__":
    main()
