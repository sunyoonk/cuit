"""Fixed set of cross-fitted probes used by IGQ."""
import torch
import torch.nn.functional as F


class ProbeBank:
    def __init__(self, depth_sets, log_pi, device="cuda"):
        self.L = len(depth_sets) - 1
        self.U, self.b, self.T, self.logW = [], [], [], []
        self.train_mask = []
        for sets in depth_sets:
            N = len(sets[0].folds)
            U, b, T, W, TM = [], [], [], [], []
            for g in sets:
                for f, p in enumerate(g.probes):
                    U.append(p.U)
                    b.append(p.b)
                    T.append(p.T)
                    w = torch.where(g.folds == f, 1.0, 0.0)
                    w = torch.where(g.folds < 0, 1.0 / g.n_folds, w)
                    W.append(w / len(sets))
                    TM.append(((g.folds >= 0) & (g.folds != f)).to(device))
            self.train_mask.append(torch.stack(TM))
            self.U.append(torch.stack(U).to(device).float())
            self.b.append(torch.stack(b).to(device).float())
            self.T.append(torch.tensor(T, device=device, dtype=torch.float32))
            W = torch.stack(W, 1).to(device)
            assert torch.allclose(W.sum(1), torch.ones(N, device=device))
            self.logW.append(W.log())
        self.log_pi = log_pi.to(device).float()

    def log_q(self, l, x, items):
        xn = x.float() / x.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
        logw = self.logW[l][items]
        acc = None
        for p in range(self.U[l].shape[0]):
            lp = F.log_softmax((xn @ self.U[l][p].T + self.b[l][p]) / self.T[l][p], -1) + logw[:, p:p + 1]
            acc = lp if acc is None else torch.logaddexp(acc, lp)
        return acc

    def heldout_violations(self, l, items):
        w = torch.isfinite(self.logW[l][items])
        return int((w & self.train_mask[l][:, items].T).sum())

    def kl_to(self, logp0, l, x, items):
        lq = self.log_q(l, x, items)
        return (logp0.exp() * (logp0 - lq)).sum(-1)

    def candidate_kl(self, logp0, l, zprev, E, items, chunk=256, y=None):
        out = []
        for s in range(0, zprev.shape[0], chunk):
            zp, it = zprev[s:s + chunk].float(), items[s:s + chunk]
            lp0 = None if logp0 is None else logp0[s:s + chunk]
            zz = (zp * zp).sum(-1, keepdim=True)
            ze = zp @ E.T
            ee = (E * E).sum(-1)
            n2 = zz + 2 * ze + ee[None]
            inv = n2.clamp_min(1e-24).rsqrt()[..., None]
            logw = self.logW[l][it]
            acc = None
            for p in range(self.U[l].shape[0]):
                lw = logw[:, p]
                if torch.isinf(lw).all():
                    continue
                uz = zp @ self.U[l][p].T
                ue = (E @ self.U[l][p].T)[None]
                lg = ((uz[:, None, :] + ue) * inv + self.b[l][p]) / self.T[l][p]
                lq = F.log_softmax(lg, -1) + lw[:, None, None]
                acc = lq if acc is None else torch.logaddexp(acc, lq)
            if y is not None:
                yy = y[s:s + chunk]
                out.append(-acc.gather(2, yy[:, None, None].expand(-1, acc.shape[1], 1)).squeeze(2))
            else:
                out.append((lp0.exp()[:, None, :] * (lp0[:, None, :] - acc)).sum(-1))
        return torch.cat(out)

    def candidate_nll(self, l, zprev, E, items, y, chunk=256):
        return self.candidate_kl(None, l, zprev, E, items, chunk, y=y)
