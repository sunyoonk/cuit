"""Generator training with validation-based checkpoint selection.

    python -m cuit.generator.train dataset=beauty seed=42
"""
import json
import os
import time

import numpy as np
import torch

from cuit import config as C
from cuit.data import amazon
from cuit.generator import beam as B
from cuit.generator.data import TrainWindows
from cuit.generator.evaluate import metrics, ranks
from cuit.generator.model import TIGER


def main():
    cfg = C.load_config()
    g, dev = cfg.gen, "cuda"
    out = os.path.join(cfg.out_dir, "generator")
    C.save_config(cfg, out, "generator")
    log = C.Logger(os.path.join(out, "train.log"))
    C.seed_all(cfg.seed, deterministic=False)
    torch.set_float32_matmul_precision(g.get("matmul_precision", "high"))
    d = amazon.load(cfg.data_dir, cfg.work_dir)
    t = torch.load(cfg.get("sid_file") or os.path.join(cfg.out_dir, "tokenizer", "sid.pt"))
    sid, path, lengths = t["sid"].long(), t["path"].long(), t["lengths"].long()
    trie = B.Trie(sid, lengths, cfg.tok.K, dev)
    model = TIGER(g, cfg.tok.K, sid.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=g.lr, weight_decay=g.weight_decay)
    loader = torch.utils.data.DataLoader(TrainWindows(d["train"], path, sid, g, cfg.seed), batch_size=None,
                                         num_workers=g.num_workers, persistent_workers=True, prefetch_factor=4)
    it = iter(loader)
    log(f"[gen] {sum(p.numel() for p in model.parameters())} parameters; mean SID length "
        f"{lengths.float().mean():.4f}")
    best, best_step, bad, hist, t0, run = -1.0, 0, 0, [], time.time(), 0.0
    for step in range(1, g.max_steps + 1):
        model.train()
        for _ in range(g.accumulate):
            h, y = next(it)
            h, y = h.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            loss = model.loss(h, y) / g.accumulate
            loss.backward()
            run += loss.item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % g.val_every:
            continue
        m = metrics(ranks(model, trie, d["valid"], path, g, dev))
        hist.append({"step": step, "loss": run / g.val_every, **m})
        run = 0.0
        if m["recall@5"] > best:
            best, best_step, bad = m["recall@5"], step, 0
            torch.save({"model": model.state_dict(), "step": step}, os.path.join(out, "best.pt"))
        else:
            bad += 1
        log(f"step {step} loss {hist[-1]['loss']:.4f} val R@5 {m['recall@5']:.4f} N@10 {m['ndcg@10']:.4f} "
            f"best {best:.4f}@{best_step} ({bad}/{g.patience}) {(time.time() - t0) / 60:.1f} min")
        if bad >= g.patience:
            break
    json.dump(hist, open(os.path.join(out, "val_history.json"), "w"), indent=1)
    model.load_state_dict(torch.load(os.path.join(out, "best.pt"))["model"])
    for split, seqs in (("valid", d["valid"]), ("test", d["test"])):
        r = ranks(model, trie, seqs, path, g, dev)
        m = metrics(r)
        np.savez(os.path.join(out, f"ranks_{split}.npz"), users=d["users"], rank=r)
        json.dump({"step": best_step, **m}, open(os.path.join(out, f"metrics_{split}.json"), "w"), indent=1)
        log(f"[{split}] best step {best_step}: " + " ".join(f"{k} {v:.4f}" for k, v in m.items()))


if __name__ == "__main__":
    main()
