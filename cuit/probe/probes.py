"""Cross-fitted linear CF probes with temperature scaling."""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F


def make_folds(y, n_folds, seed):
    rng = np.random.default_rng(seed)
    lab = np.flatnonzero(y >= 0)
    perm = rng.permutation(lab)
    folds = np.full(len(y), -1, np.int64)
    folds[perm] = np.arange(len(perm)) % n_folds
    return folds


def _normalize(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


@dataclass
class LinearProbe:
    U: torch.Tensor
    b: torch.Tensor
    T: float = 1.0

    def logits(self, x, use_T=True):
        z = _normalize(x.float()) @ self.U.T + self.b
        return z / self.T if use_T else z

    def log_probs(self, x, use_T=True):
        return F.log_softmax(self.logits(x, use_T), dim=-1)

    def state(self):
        return {"U": self.U.cpu(), "b": self.b.cpu(), "T": self.T}


def fit_linear(x, y, M, lambda_U, max_iter=500, history=20):
    xn = _normalize(x.float())
    U = torch.zeros(M, xn.shape[1], device=xn.device, requires_grad=True)
    b = torch.zeros(M, device=xn.device, requires_grad=True)
    opt = torch.optim.LBFGS([U, b], lr=1.0, max_iter=max_iter, history_size=history,
                            tolerance_grad=1e-6, tolerance_change=1e-10, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(xn @ U.T + b, y, reduction="sum") + lambda_U * U.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return U.detach(), b.detach()


def fit_temperature(logits, y):
    logT = torch.zeros(1, device=logits.device, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.5, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / logT.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(logT.exp())


@dataclass
class CrossFitProbe:
    probes: List[LinearProbe]
    log_pi_fold: torch.Tensor
    log_pi_all: torch.Tensor
    folds: torch.Tensor
    info: dict = field(default_factory=dict)

    @property
    def n_folds(self):
        return len(self.probes)

    def log_probs(self, x, item_idx, use_T=True):
        fo = self.folds[item_idx]
        out = torch.empty(x.shape[0], self.log_pi_all.shape[0], device=x.device)
        for f, p in enumerate(self.probes):
            m = fo == f
            if m.any():
                out[m] = p.log_probs(x[m], use_T)
        m = fo < 0
        if m.any():
            lp = torch.stack([p.log_probs(x[m], use_T) for p in self.probes])
            out[m] = torch.logsumexp(lp, 0) - np.log(self.n_folds)
        return out

    def log_pi(self, item_idx):
        fo = self.folds[item_idx]
        out = self.log_pi_all.expand(len(item_idx), -1).clone()
        lab = fo >= 0
        out[lab] = self.log_pi_fold[fo[lab]]
        return out

    def state(self):
        return {"probes": [p.state() for p in self.probes], "log_pi_fold": self.log_pi_fold.cpu(),
                "log_pi_all": self.log_pi_all.cpu(), "folds": self.folds.cpu(), "info": self.info}

    @staticmethod
    def from_state(s, device):
        return CrossFitProbe([LinearProbe(p["U"].to(device), p["b"].to(device), p["T"]) for p in s["probes"]],
                             s["log_pi_fold"].to(device), s["log_pi_all"].to(device), s["folds"].to(device),
                             s.get("info", {}))


def label_marginal(y_sub, M):
    cnt = torch.bincount(y_sub, minlength=M).float()
    assert (cnt > 0).all(), f"label marginal has empty groups: {(cnt == 0).sum().item()} of {M}"
    return (cnt / cnt.sum()).log()


def fit_crossfit(x_rows, row_item, y, folds, M, pcfg, seed, lam, log=print, name=""):
    dev = x_rows.device
    n_folds = int(folds.max()) + 1
    lab_items = torch.nonzero(y >= 0).squeeze(1)
    log_pi_fold = []
    probes = []
    info = {"name": name, "lambda_U": float(lam), "folds": []}
    row_fold = folds[row_item]
    row_y = y[row_item]
    for f in range(n_folds):
        tr_items = lab_items[folds[lab_items] != f]
        assert not (folds[tr_items] == f).any()
        log_pi_fold.append(label_marginal(y[tr_items], M))
        g = torch.Generator().manual_seed(seed * 100 + f)
        hold = torch.zeros(len(y), dtype=torch.bool)
        perm = tr_items.cpu()[torch.randperm(len(tr_items), generator=g)]
        hold[perm[: int(round(pcfg.temp_frac * len(perm)))]] = True
        hold = hold.to(dev)
        rows_tr = (row_fold >= 0) & (row_fold != f) & ~hold[row_item]
        rows_T = (row_fold >= 0) & (row_fold != f) & hold[row_item]
        U, b = fit_linear(x_rows[rows_tr], row_y[rows_tr], M, lam, pcfg.lbfgs_max_iter, pcfg.lbfgs_history)
        p = LinearProbe(U, b, 1.0)
        rec = {"fold": f, "n_fit": int(rows_tr.sum()), "n_temp": int(rows_T.sum())}
        p.T = fit_temperature(p.logits(x_rows[rows_T], use_T=False), row_y[rows_T])
        rows_te = row_fold == f
        lg = p.logits(x_rows[rows_te])
        rec.update({"T": p.T, "heldout_acc": float((lg.argmax(1) == row_y[rows_te]).float().mean()),
                    "heldout_nll": float(F.cross_entropy(lg, row_y[rows_te])),
                    "fit_acc": float((p.logits(x_rows[rows_tr]).argmax(1) == row_y[rows_tr]).float().mean())})
        info["folds"].append(rec)
        probes.append(p)
        log(f"  [{name}] fold {f}: fit={rec['n_fit']} temp={rec['n_temp']} T={p.T:.3f} "
            f"fit acc={rec['fit_acc']:.3f} held-out acc={rec['heldout_acc']:.3f} nll={rec['heldout_nll']:.3f}")
    log_pi_all = label_marginal(y[lab_items], M)
    n_te = [((row_fold == r["fold"])).sum().item() for r in info["folds"]]
    info["heldout_nll"] = float(sum(r["heldout_nll"] * n for r, n in zip(info["folds"], n_te)) / sum(n_te))
    info["heldout_acc"] = float(sum(r["heldout_acc"] * n for r, n in zip(info["folds"], n_te)) / sum(n_te))
    return CrossFitProbe(probes, torch.stack(log_pi_fold), log_pi_all, folds, info)


def fit_crossfit_select(x_rows, row_item, y, folds, M, pcfg, seed, log=print, name=""):
    grid = list(pcfg.lambda_grid)
    best, table = None, []
    for lam in grid:
        g = fit_crossfit(x_rows, row_item, y, folds, M, pcfg, seed, lam, lambda *a: None, name)
        table.append({"lambda_U": float(lam), "heldout_nll": g.info["heldout_nll"], "heldout_acc": g.info["heldout_acc"]})
        log(f"  [{name}] lambda_U={lam:g}: held-out nll={g.info['heldout_nll']:.4f} acc={g.info['heldout_acc']:.4f}")
        if best is None or g.info["heldout_nll"] < best.info["heldout_nll"]:
            best = g
    best.info["lambda_selection"] = table
    log(f"  [{name}] selected lambda_U={best.info['lambda_U']:g} (held-out nll={best.info['heldout_nll']:.4f}); "
        f"T per fold={[round(r['T'], 3) for r in best.info['folds']]}")
    return best
