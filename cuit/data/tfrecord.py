"""Minimal reader for GZIP TFRecord files of tf.train.Example (no TensorFlow dependency)."""
import glob
import gzip
import os
import struct


def _varint(buf, i):
    out, shift = 0, 0
    while True:
        b = buf[i]
        i += 1
        out |= (b & 0x7F) << shift
        if b < 0x80:
            return out, i
        shift += 7


def _fields(buf):
    i, n = 0, len(buf)
    while i < n:
        key, i = _varint(buf, i)
        f, w = key >> 3, key & 7
        if w == 0:
            v, i = _varint(buf, i)
        elif w == 1:
            v, i = buf[i:i + 8], i + 8
        elif w == 2:
            ln, i = _varint(buf, i)
            v, i = buf[i:i + ln], i + ln
        elif w == 5:
            v, i = buf[i:i + 4], i + 4
        else:
            raise ValueError(f"unsupported wire type {w}")
        yield f, w, v


def _feature(buf):
    for kind, _, lst in _fields(buf):
        vals = []
        for _, w, v in _fields(lst):
            if kind == 1:
                vals.append(bytes(v))
            elif kind == 2:
                vals.extend(struct.unpack(f"<{len(v) // 4}f", v) if w == 2 else struct.unpack("<f", v))
            elif kind == 3:
                if w == 2:
                    j = 0
                    while j < len(v):
                        x, j = _varint(v, j)
                        vals.append(x - (1 << 64) if x >= 1 << 63 else x)
                else:
                    vals.append(v - (1 << 64) if v >= 1 << 63 else v)
        return vals
    return []


def parse_example(buf, keys=None):
    out = {}
    for _, _, features in _fields(buf):
        for _, _, entry in _fields(features):
            name, value = None, None
            for f, _, v in _fields(entry):
                if f == 1:
                    name = bytes(v).decode()
                else:
                    value = v
            if keys is None or name in keys:
                out[name] = _feature(value)
    return out


def read_records(folder, keys=None):
    files = sorted(glob.glob(os.path.join(folder, "*.tfrecord.gz")))
    assert files, f"no TFRecord files in {folder}"
    for path in files:
        with gzip.open(path, "rb") as f:
            data = f.read()
        i = 0
        while i < len(data):
            (ln,) = struct.unpack("<Q", data[i:i + 8])
            i += 12
            yield parse_example(memoryview(data)[i:i + ln], keys)
            i += ln + 4
