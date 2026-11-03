"""LightGCN on the training interactions."""
import numpy as np
import torch


class LightGCN(torch.nn.Module):
    def __init__(self, num_users, num_items, adj, dim=64, num_layers=3):
        super().__init__()
        self.num_users, self.num_items, self.num_layers, self.adj = num_users, num_items, num_layers, adj
        self.emb = torch.nn.Embedding(num_users + num_items, dim)
        torch.nn.init.normal_(self.emb.weight, std=0.1)

    def propagate(self):
        e = self.emb.weight
        out = [e]
        for _ in range(self.num_layers):
            e = torch.sparse.mm(self.adj, e)
            out.append(e)
        e = torch.stack(out).mean(0)
        return e[: self.num_users], e[self.num_users:]


def norm_adj(users, items, num_users, num_items, device):
    n = num_users + num_items
    rows = np.r_[users, items + num_users]
    cols = np.r_[items + num_users, users]
    deg = np.bincount(rows, minlength=n).astype(np.float64)
    vals = 1.0 / np.sqrt(deg[rows] * deg[cols])
    idx = torch.tensor(np.stack([rows, cols]), dtype=torch.long)
    return torch.sparse_coo_tensor(idx, torch.tensor(vals, dtype=torch.float32), (n, n)).coalesce().to(device)


@torch.no_grad()
def recall_at_20(model, train_pos, valid_items, device):
    u_emb, i_emb = model.propagate()
    hits = 0
    for b in range(0, len(train_pos), 4096):
        scores = u_emb[b: b + 4096] @ i_emb.T
        for r in range(scores.shape[0]):
            scores[r, torch.as_tensor(train_pos[b + r], device=device)] = -float("inf")
        top = scores.topk(20, dim=1).indices.cpu().numpy()
        hits += (top == valid_items[b: b + 4096, None]).any(1).sum()
    return hits / len(train_pos)


def _fit(pos_lists, num_items, device, cfg, epochs, valid_items=None, log=print):
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    num_users = len(pos_lists)
    pos_lists = [np.unique(p) for p in pos_lists]
    users = np.concatenate([np.full(len(p), u) for u, p in enumerate(pos_lists)])
    items = np.concatenate(pos_lists)
    pos_sets = [set(p.tolist()) for p in pos_lists]
    adj = norm_adj(users, items, num_users, num_items, device)
    model = LightGCN(num_users, num_items, adj, cfg.dim, cfg.layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    best, best_epoch, bad, history = -1.0, 0, 0, []
    for epoch in range(epochs):
        order = rng.permutation(len(users))
        total = 0.0
        for b in range(0, len(order), cfg.batch_size):
            idx = order[b: b + cfg.batch_size]
            u, p = users[idx], items[idx]
            n = rng.integers(0, num_items, len(idx))
            for j in range(len(n)):
                while n[j] in pos_sets[u[j]]:
                    n[j] = rng.integers(0, num_items)
            u_t, p_t, n_t = (torch.as_tensor(a, device=device) for a in (u, p, n))
            u_emb, i_emb = model.propagate()
            x_u, x_p, x_n = u_emb[u_t], i_emb[p_t], i_emb[n_t]
            bpr = -torch.nn.functional.logsigmoid((x_u * x_p).sum(1) - (x_u * x_n).sum(1)).mean()
            e0 = model.emb.weight
            l2 = 0.5 * (e0[u_t].pow(2).sum() + e0[num_users + p_t].pow(2).sum()
                        + e0[num_users + n_t].pow(2).sum()) / len(idx)
            loss = bpr + cfg.reg * l2
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if valid_items is None:
            if epoch % 20 == 0:
                log(f"  refit epoch {epoch:4d} loss={total / len(users):.4f}")
            continue
        rec = float(recall_at_20(model, pos_lists, valid_items, device))
        history.append({"epoch": epoch + 1, "loss": total / len(users), "heldout_recall20": rec})
        if rec > best:
            best, best_epoch, bad = rec, epoch + 1, 0
        else:
            bad += 1
        if epoch % 10 == 0 or bad == 0:
            log(f"  epoch {epoch + 1:4d} loss={total / len(users):.4f} held-out R@20={rec:.4f} best={best:.4f}@{best_epoch}")
        if bad >= cfg.patience:
            break
    return model, best, best_epoch, history


def train_lightgcn(train_seqs, num_items, device, cfg, log=print):
    sel_pos = [s[:-1] for s in train_seqs]
    sel_valid = np.array([s[-1] for s in train_seqs])
    log("[lightgcn] epoch selection on held-out last train item")
    _, best, best_epoch, history = _fit(sel_pos, num_items, device, cfg, cfg.max_epochs, sel_valid, log)
    log(f"[lightgcn] refit on full train split for {best_epoch} epochs")
    model, _, _, _ = _fit(list(train_seqs), num_items, device, cfg, best_epoch, None, log)
    model.eval()
    with torch.no_grad():
        _, i_emb = model.propagate()
    return i_emb.cpu().numpy(), {"heldout_recall20": best, "epochs": best_epoch, "history": history}
