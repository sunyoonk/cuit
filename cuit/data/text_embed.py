"""Item text embeddings (Flan-T5-XL encoder, mean pooling).

    python -m cuit.data.text_embed dataset=beauty
"""
import os

import torch
from transformers import AutoTokenizer, T5EncoderModel

from cuit import config as C
from cuit.data import amazon


@torch.no_grad()
def embed(texts, model_name, max_length, batch_size, device):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = T5EncoderModel.from_pretrained(model_name).to(device).eval()
    out = []
    for s in range(0, len(texts), batch_size):
        b = tok(texts[s:s + batch_size], max_length=max_length, padding="max_length", truncation=True,
                return_tensors="pt").to(device)
        h = model(input_ids=b.input_ids, attention_mask=b.attention_mask).last_hidden_state
        m = b.attention_mask[..., None].float()
        out.append(((h * m).sum(1) / m.sum(1)).float().cpu())
    return torch.cat(out)


def main():
    cfg = C.load_config()
    path = os.path.join(cfg.work_dir, "text_emb.pt")
    if os.path.exists(path):
        print(f"{path} exists")
        return
    d = amazon.load(cfg.data_dir, cfg.work_dir)
    z = embed(d["texts"], cfg.embed.model, cfg.embed.max_length, cfg.embed.batch_size, "cuda")
    torch.save(z, path)
    print(f"{tuple(z.shape)} -> {path}")


if __name__ == "__main__":
    main()
