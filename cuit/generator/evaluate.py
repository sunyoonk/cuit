"""Recall@N / NDCG@N over all items.

    python -m cuit.generator.evaluate dataset=beauty seed=42 split=test
"""
import json
import os

import numpy as np
import torch

from cuit import config as C
from cuit.data import amazon
from cuit.generator import beam as B
from cuit.generator.data import eval_batches
from cuit.generator.model import TIGER


@torch.no_grad()
def ranks(model, trie, seqs, path, g, device):
    model.eval()
    out = np.full(len(seqs), 10, np.int64)
    for users, hist, tgt in eval_batches(seqs, path, g.eval_batch_size):
        ids, _ = B.generate(model, trie, hist.to(device), g.beam, 10)
        hit = ids == tgt.to(device)[:, None]
        r = torch.where(hit.any(1), hit.float().argmax(1), torch.full_like(tgt.to(device), 10))
        out[users] = r.cpu().numpy()
    return out


def metrics(r):
    out = {}
    for n in (5, 10):
        hit = r < n
        out[f"recall@{n}"] = float(hit.mean())
        out[f"ndcg@{n}"] = float(np.where(hit, 1.0 / np.log2(r + 2.0), 0.0).mean())
    return out


def main():
    cfg = C.load_config()
    split = cfg.get("split", "test")
    g, dev = cfg.gen, "cuda"
    torch.set_float32_matmul_precision(g.get("matmul_precision", "high"))
    d = amazon.load(cfg.data_dir, cfg.work_dir)
    t = torch.load(cfg.get("sid_file") or os.path.join(cfg.out_dir, "tokenizer", "sid.pt"))
    sid, path, lengths = t["sid"], t["path"], t["lengths"]
    model = TIGER(g, cfg.tok.K, sid.shape[1]).to(dev)
    model.load_state_dict(torch.load(cfg.get("ckpt") or os.path.join(cfg.out_dir, "generator", "best.pt"))["model"])
    trie = B.Trie(sid, lengths, cfg.tok.K, dev)
    r = ranks(model, trie, d["valid" if split == "valid" else "test"], path, g, dev)
    m = metrics(r)
    out = cfg.get("eval_dir") or os.path.join(cfg.out_dir, "generator")
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, f"ranks_{split}.npz"), users=d["users"], rank=r)
    json.dump(m, open(os.path.join(out, f"metrics_{split}.json"), "w"), indent=1)
    print(json.dumps({"split": split, **m}))


if __name__ == "__main__":
    main()
