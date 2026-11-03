"""RQ-Kmeans baseline: RQ-Kmeans codes plus a disambiguation token.

    python -m cuit.tokenizer.rq_baseline dataset=beauty seed=42
"""
import os

import torch

from cuit import config as C
from cuit.tokenizer import rq


def main():
    cfg = C.load_config()
    out = os.path.join(cfg.out_dir, "rq_baseline")
    C.save_config(cfg, out, "rq_baseline")
    C.seed_all(cfg.seed)
    t = cfg.tok
    z = torch.load(os.path.join(cfg.work_dir, "text_emb.pt")).float().cuda()
    codes, cents, scale = rq.rq_kmeans(z, t.K, t.L_max, t.kmeans, cfg.seed)
    inv = torch.unique(codes, dim=0, return_inverse=True)[1].cpu()
    dedup = torch.zeros(len(z), dtype=torch.long)
    seen = {}
    for i, g in enumerate(inv.tolist()):
        dedup[i] = seen.get(g, 0)
        seen[g] = dedup[i].item() + 1
    assert int(dedup.max()) < t.K
    sid = torch.cat([codes.cpu(), dedup[:, None]], 1)
    torch.save({"sid": sid, "path": sid.clone(), "lengths": torch.full((len(z),), t.L_max + 1), "K": t.K},
               os.path.join(out, "sid.pt"))
    print(f"{len(torch.unique(codes, dim=0))} distinct code paths, max disambiguation index {int(dedup.max())}")


if __name__ == "__main__":
    main()
