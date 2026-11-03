"""TIGER-style encoder-decoder generator."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.t5.modeling_t5 import T5Config, T5LayerFF, T5LayerNorm, T5Stack


class _MLP(nn.Module):
    def __init__(self, d, d_ff, n_hidden, dropout):
        super().__init__()
        dims = [d] + [d_ff] * n_hidden + [d]
        layers = [nn.Linear(dims[0], dims[1])]
        for a, b in zip(dims[1:-1], dims[2:]):
            layers += [nn.ReLU(), nn.Linear(a, b), nn.Dropout(dropout)]
        self.model = nn.Sequential(*layers)


class _FF(nn.Module):
    def __init__(self, cfg, n_hidden):
        super().__init__()
        self.mlp = _MLP(cfg.d_model, cfg.d_ff, n_hidden, cfg.dropout_rate)
        self.layer_norm = T5LayerNorm(cfg.d_model, eps=cfg.layer_norm_epsilon)
        self.dropout = nn.Dropout(cfg.dropout_rate)

    def forward(self, h):
        return h + self.dropout(self.mlp.model(self.layer_norm(h)))


def _reset(m):
    if hasattr(m, "reset_parameters"):
        m.reset_parameters()
    else:
        for c in m.children():
            _reset(c)


def _stack(g, layers, decoder):
    cfg = T5Config(vocab_size=1, d_model=g.d_model, d_ff=g.d_ff, num_heads=g.heads, d_kv=g.d_kv, num_layers=layers,
                   dropout_rate=g.dropout, is_decoder=decoder, is_encoder_decoder=False, use_cache=False)
    s = T5Stack(cfg)
    del s.embed_tokens
    _reset(s)
    for blk in s.block:
        for i, layer in enumerate(blk.layer):
            if isinstance(layer, T5LayerFF):
                blk.layer[i] = _FF(cfg, g.ff_layers)
    return s


class TIGER(nn.Module):
    def __init__(self, g, K, L):
        super().__init__()
        self.K, self.L, self.V = K, L, K + 1
        self.encoder = _stack(g, g.enc_layers, False)
        self.decoder = _stack(g, g.dec_layers, True)
        self.bos = nn.Parameter(torch.randn(1, g.d_model))
        self.heads = nn.ModuleList([nn.Linear(g.d_model, self.V, bias=False) for _ in range(L)])
        self.emb = nn.Embedding(self.V * L, g.d_model)
        self.sep = nn.Parameter(torch.randn(1, g.d_model))

    def _tok(self, ids):
        off = torch.arange(ids.shape[-1], device=ids.device) * self.V
        return self.emb(ids.clamp_min(0) + off)

    def encode(self, hist):
        B, T, L = hist.shape
        mask = (hist[..., 0] >= 0)
        x = torch.cat([self._tok(hist), self.sep.expand(B, T, 1, -1)], 2).reshape(B, T * (L + 1), -1)
        m = mask[..., None].expand(B, T, L + 1).reshape(B, -1).long()
        return self.encoder(inputs_embeds=x, attention_mask=m).last_hidden_state, m

    def decode(self, prefix, enc, enc_mask):
        x = self.bos.expand(len(enc), 1, -1)
        if prefix.shape[1]:
            x = torch.cat([x, self._tok(prefix)], 1)
        return self.decoder(inputs_embeds=x, encoder_hidden_states=enc, encoder_attention_mask=enc_mask).last_hidden_state

    def loss(self, hist, target):
        enc, m = self.encode(hist)
        h = self.decode(target[:, :-1], enc, m)
        return sum(F.cross_entropy(self.heads[l](h[:, l]), target[:, l]) for l in range(self.L))
