"""Amazon 2014 user sequences (leave-last-out split)."""
import os
import pickle

import numpy as np

from cuit.data.tfrecord import read_records


def load(data_dir, work_dir):
    cache = os.path.join(work_dir, "sequences.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)

    def seqs(split):
        return {r["user_id"][0]: np.array(r["sequence_data"], np.int64)
                for r in read_records(os.path.join(data_dir, split), ("user_id", "sequence_data"))}

    tr, va, te = seqs("training"), seqs("evaluation"), seqs("testing")
    assert tr.keys() == va.keys() == te.keys()
    users = sorted(tr)
    texts = {r["id"][0]: r["text"][0].decode() for r in read_records(os.path.join(data_dir, "items"), ("id", "text"))}
    n_items = len(texts)
    assert sorted(texts) == list(range(n_items))
    d = {"users": np.array(users), "train": [tr[u] for u in users], "valid": [va[u] for u in users],
         "test": [te[u] for u in users], "texts": [texts[i] for i in range(n_items)], "num_items": n_items}
    for u in range(len(users)):
        full = np.concatenate([d["train"][u], d["test"][u][-2:]])
        assert np.array_equal(full[:-1][-len(d["valid"][u]):], d["valid"][u])
        assert np.array_equal(full[-len(d["test"][u]):], d["test"][u])
    os.makedirs(work_dir, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(d, f)
    return d
