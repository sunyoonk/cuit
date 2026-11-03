"""CF labels: LightGCN item embeddings clustered by k-means (-1 for items without training interactions).

    python -m cuit.cf.labels dataset=beauty
"""
import json
import os

import numpy as np
import torch
from sklearn.cluster import KMeans

from cuit import config as C
from cuit.cf.lightgcn import train_lightgcn
from cuit.data import amazon


def main():
    cfg = C.load_config()
    out = os.path.join(cfg.work_dir, "cf")
    if os.path.exists(os.path.join(out, "y.npy")):
        print(f"{out}/y.npy exists")
        return
    log = C.Logger(os.path.join(out, "cf.log"))
    C.seed_all(cfg.cf.seed)
    d = amazon.load(cfg.data_dir, cfg.work_dir)
    n = d["num_items"]
    count = np.zeros(n, np.int64)
    for s in d["train"]:
        np.add.at(count, s, 1)
    labeled = count > 0
    log(f"[cf] users={len(d['train'])} items={n} train interactions={count.sum()} labeled items={labeled.sum()}")
    emb, info = train_lightgcn(d["train"], n, "cuda", cfg.cf, log)
    np.save(os.path.join(out, "lightgcn_item_emb.npy"), emb)
    c = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    km = KMeans(n_clusters=cfg.cf.M, init="k-means++", n_init=1, random_state=cfg.cf.kmeans_seed).fit(c[labeled])
    y = np.full(n, -1, np.int64)
    y[labeled] = km.labels_
    np.save(os.path.join(out, "y.npy"), y)
    info.update({"M": int(cfg.cf.M), "labeled": int(labeled.sum())})
    json.dump(info, open(os.path.join(out, "lightgcn.json"), "w"), indent=1)
    log(f"[cf] held-out R@20={info['heldout_recall20']:.4f} epochs={info['epochs']}")


if __name__ == "__main__":
    main()
