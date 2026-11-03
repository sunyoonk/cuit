"""Information-Guided Quantization (one round, probes fixed)."""
import numpy as np
import torch
import torch.nn.functional as F

from cuit.probe import cip
from cuit.tokenizer import kmeans
from cuit.tokenizer.rq import kmeans_seed

INFO, IDENT, STOP = 0, 1, 2


def _greedy_distinct(cost, taken):
    cost = cost.copy()
    cost[:, sorted(taken)] = np.inf
    order = np.argsort(cost.min(1), kind="stable")
    out = np.full(len(cost), -1, np.int64)
    used = np.zeros(cost.shape[1], bool)
    for i in order:
        row = np.where(used, np.inf, cost[i])
        k = int(row.argmin())
        if np.isfinite(row[k]):
            out[i] = k
            used[k] = True
    return out


def _groups(prefix):
    if prefix.shape[1] == 0:
        return torch.zeros(len(prefix), dtype=torch.long, device=prefix.device)
    return torch.unique(prefix, dim=0, return_inverse=True)[1]


@torch.no_grad()
def igq_round(z, y, bank, logp0, A, A0, cfg, log):
    t = cfg.tok
    K, L, lam, dev = t.K, t.L_max, t.lam, z.device
    N = len(z)
    lab = y >= 0
    nlogpi = -bank.log_pi.gather(1, y.clamp_min(0)[:, None]).squeeze(1)
    r, cum = z.float().clone(), z.float().norm(dim=1)
    zhat = torch.zeros_like(r)
    codes = torch.full((N, L), K, dtype=torch.long, device=dev)
    lengths = torch.full((N,), L, dtype=torch.long, device=dev)
    mode = torch.where(A <= t.delta, IDENT, INFO).long()
    floor = torch.zeros(N, dtype=torch.bool, device=dev)
    rho_prev = torch.ones(N, device=dev)
    cents, scale, stats = [], [], []
    R_all = torch.zeros(L, N, device=dev)
    rho_all = torch.zeros(L, N, device=dev)
    mode_hist = [mode.clone()]

    for l in range(L):
        act = torch.nonzero((mode != STOP) | floor).squeeze(1)
        rb = F.normalize(r[act], dim=1)
        C = kmeans.fit(rb, K, t.kmeans, kmeans_seed(cfg.seed, l))
        D = kmeans.sq_dist(rb, C).clamp_min(0)
        s_l = float(cum[act].mean())
        sigma2 = float(D.min(1).values.mean())
        E = s_l * C
        m = mode[act]
        nearest = D.argmin(1)
        code = nearest.clone()

        # Info items
        info = torch.nonzero((m == INFO) & ~floor[act]).squeeze(1)
        score9 = D / sigma2
        if len(info):
            it = act[info]
            kl = bank.candidate_kl(logp0[it], l + 1, zhat[it], E, it)
            score9[info] = score9[info] + lam * rho_prev[it, None] * kl
            code[info] = score9[info].argmin(1)
        n_merged = 0
        if l == 0 and len(info):
            cnt = torch.bincount(code[info], minlength=K)
            alone = info[cnt[code[info]] == 1]
            if len(alone):
                d = D[alone].masked_fill((cnt < 2)[None], float("inf"))
                code[alone] = d.argmin(1)
                n_merged = len(alone)

        # Ident items
        grp = _groups(codes[act, :l])
        ident = torch.nonzero((m == IDENT) & ~floor[act]).squeeze(1)
        n_ident_free = 0
        if len(ident):
            it = act[ident]
            c13 = D[ident] / sigma2
            li = lab[it]
            if li.any():
                nll = bank.candidate_nll(l + 1, zhat[it[li]], E, it[li], y[it[li]])
                c13[li] = c13[li] + lam * nll / nlogpi[it[li], None]
            if (~li).any():
                c13[~li] = c13[~li] + lam * bank.candidate_kl(logp0[it[~li]], l + 1, zhat[it[~li]], E, it[~li])
            g_np, c_np = grp.cpu().numpy(), code.cpu().numpy()
            info_mask = np.zeros(len(act), bool)
            info_mask[info.cpu().numpy()] = True
            ident_np, c13_np = ident.cpu().numpy(), c13.cpu().numpy()
            for g in np.unique(g_np[ident_np]):
                rows = np.flatnonzero(g_np[ident_np] == g)
                taken = set(c_np[info_mask & (g_np == g)].tolist())
                pick = _greedy_distinct(c13_np[rows], taken)
                ok = pick >= 0
                c_np[ident_np[rows[ok]]] = pick[ok]
                n_ident_free += int(ok.sum())
            code = torch.as_tensor(c_np, device=dev)

        # last depth: make SIDs unique
        n_resolved = 0
        if l == L - 1:
            full = torch.cat([codes[act, :l], code[:, None]], 1)
            inv, cnt = torch.unique(full, dim=0, return_inverse=True, return_counts=True)[1:]
            clash = cnt[inv] > 1
            if clash.any():
                g_np, c_np, s9 = grp.cpu().numpy(), code.cpu().numpy(), score9.cpu().numpy()
                clash_np = clash.cpu().numpy()
                for g in np.unique(g_np[clash_np]):
                    rows = np.flatnonzero((g_np == g) & clash_np)
                    taken = set(c_np[(g_np == g) & ~clash_np].tolist())
                    pick = _greedy_distinct(s9[rows], taken)
                    assert (pick >= 0).all(), "a prefix is shared by more than K items"
                    c_np[rows] = pick
                    n_resolved += len(rows)
                code = torch.as_tensor(c_np, device=dev)

        codes[act, l] = code
        r[act] = rb - C[code]
        cum[act] = cum[act] * r[act].norm(dim=1)
        zhat[act] = zhat[act] + E[code]
        R = torch.cat([bank.kl_to(logp0[act[s:s + 4096]], l + 1, zhat[act[s:s + 4096]], act[s:s + 4096])
                       for s in range(0, len(act), 4096)]).clamp_min(0)
        R_all[l, act] = R
        rho_all[l, act] = cip.rho(A[act], R, A0)

        # Info / Ident / Stop
        shared = torch.zeros(N, dtype=torch.bool, device=dev)
        inv, cnt = torch.unique(codes[act, :l + 1], dim=0, return_inverse=True, return_counts=True)[1:]
        shared[act] = cnt[inv] > 1
        new = torch.full((len(act),), STOP, dtype=torch.long, device=dev)
        if l < L - 1:
            keep_info = (mode[act] == INFO) & (rho_all[l, act] > t.eps)
            new = torch.where(keep_info, INFO, torch.where(shared[act], IDENT, STOP))
        mode[act] = new
        floor[:] = False
        if l + 1 < t.min_len:
            floor[act[new == STOP]] = True
        lengths[act[(new == STOP) & ~floor[act]]] = l + 1
        rho_prev = rho_all[l].clone()
        mode_hist.append(mode.clone())
        cents.append(C)
        scale.append(s_l)
        st = {"depth": l + 1, "active": len(act), "info": int(len(info)), "ident": int(len(ident)),
              "ident_free": n_ident_free, "floor": int(floor.sum()), "merged": n_merged, "resolved": n_resolved,
              "sigma2": sigma2, "s": s_l, "mean_R": float(R.mean()),
              "next": [int((mode == x).sum()) for x in (INFO, IDENT, STOP)]}
        stats.append(st)
        log(f"  depth {l + 1}: {st}")

    on = torch.arange(L, device=dev)[None] < lengths[:, None]
    codes = torch.where(on, codes, K)
    return {"codes": codes, "lengths": lengths, "centroids": torch.stack(cents), "scale": torch.tensor(scale),
            "R": R_all, "rho": rho_all, "mode_hist": torch.stack(mode_hist), "stats": stats}
