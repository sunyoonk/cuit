"""Constrained beam search over the trie of valid variable-length SIDs."""
import torch


class Trie:
    def __init__(self, sid, lengths, K, device):
        self.K, self.V, self.L = K, K + 1, sid.shape[1]
        sid, lengths = sid.to(device).long(), lengths.to(device).long()
        self.prefix = []
        self.leaf = []
        for l in range(self.L):
            has = lengths > l
            self.prefix.append(torch.unique(self.key(sid[has, :l + 1])))
            self.leaf.append(torch.unique(self.key(sid[lengths == l + 1, :l + 1])))
        item_key = self.key(torch.where(sid == K, 0, sid))
        self.item_key, order = torch.sort(item_key)
        self.item_of_key = order

    def key(self, rows):
        w = self.V ** torch.arange(rows.shape[-1] - 1, -1, -1, device=rows.device)
        return (rows.long() * w).sum(-1)

    @staticmethod
    def member(keys, table):
        if len(table) == 0:
            return torch.zeros_like(keys, dtype=torch.bool)
        pos = torch.searchsorted(table, keys).clamp(max=len(table) - 1)
        return table[pos] == keys

    def items(self, seqs):
        k = self.key(torch.where(seqs == self.K, 0, seqs))
        pos = torch.searchsorted(self.item_key, k).clamp(max=len(self.item_key) - 1)
        return torch.where(self.item_key[pos] == k, self.item_of_key[pos], -1)


@torch.no_grad()
def generate(model, trie, hist, beam=10, n_out=10):
    B, V, L, dev = hist.shape[0], trie.V, trie.L, hist.device
    enc, m = model.encode(hist)
    seqs = torch.zeros(B, 1, 0, dtype=torch.long, device=dev)
    prob = torch.ones(B, 1, device=dev)
    fin_s, fin_p = [], []
    codes = torch.arange(V, device=dev)
    for l in range(L):
        nb = seqs.shape[1]
        flat = seqs.reshape(B * nb, l)
        h = model.decode(flat, enc.repeat_interleave(nb, 0), m.repeat_interleave(nb, 0))[:, -1]
        logits = model.heads[l](h).reshape(B, nb, V)
        pk = trie.key(flat).reshape(B, nb, 1) * V + codes
        valid = trie.member(pk, trie.prefix[l])
        p = torch.softmax(logits.masked_fill(~valid, float("-inf")), -1)
        p = torch.nan_to_num(p, nan=0.0)
        cand = (prob[..., None] * p).reshape(B, nb * V)
        valid = valid.reshape(B, nb * V)
        nxt = torch.cat([flat.reshape(B, nb, 1, l).expand(B, nb, V, l), codes.view(1, 1, V, 1).expand(B, nb, V, 1)],
                        -1).reshape(B, nb * V, l + 1)
        if l < L - 1:
            leaf = trie.member(pk.reshape(B, nb * V), trie.leaf[l]) & valid
            if leaf.any():
                v, j = cand.masked_fill(~leaf, -1.0).topk(min(n_out, nb * V), 1)
                s = nxt.gather(1, j[..., None].expand(-1, -1, l + 1))
                fin_s.append(torch.cat([s, torch.full((B, s.shape[1], L - l - 1), trie.K, device=dev)], -1))
                fin_p.append(torch.where(v > 0, v, torch.full_like(v, -1.0)))
                cand = cand.masked_fill(leaf, -1.0)
        cand = cand.masked_fill(~valid, -1.0)
        v, j = cand.topk(min(beam, nb * V), 1)
        seqs = nxt.gather(1, j[..., None].expand(-1, -1, l + 1))
        prob = v
    all_s = torch.cat([seqs] + fin_s, 1)
    all_p = torch.cat([prob] + fin_p, 1)
    order = torch.sort(all_p, dim=1, descending=True, stable=True).indices[:, :n_out]
    top_s = all_s.gather(1, order[..., None].expand(-1, -1, L))
    top_p = all_p.gather(1, order)
    ids = trie.items(top_s)
    return torch.where(top_p > 0, ids, -1), top_p
