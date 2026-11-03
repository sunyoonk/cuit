import os
import random
import sys
import time

import numpy as np
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(argv=None):
    cfg = OmegaConf.load(os.path.join(ROOT, "configs", "cuit.yaml"))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(sys.argv[1:] if argv is None else argv)))
    for k in ("data_dir", "work_dir", "out_dir"):
        if not os.path.isabs(cfg[k]):
            cfg[k] = os.path.join(ROOT, cfg[k])
    return cfg


def save_config(cfg, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(out_dir, f"{name}_config.yaml"), resolve=True)


def seed_all(seed, deterministic=True):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


class Logger:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "a")

    def __call__(self, *a):
        msg = f"[{time.strftime('%m-%d %H:%M:%S')}] " + " ".join(str(x) for x in a)
        print(msg, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()
