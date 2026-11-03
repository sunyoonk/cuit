# CUIT

## Setup

```bash
pip install -r requirements.txt     # Python 3.10, PyTorch 2.6 (CUDA 12.4)
```

## Run

```bash
bash scripts/1_data.sh beauty          # 1. dataset download + item text embeddings
bash scripts/2_tokenize.sh beauty 42   # 2. CUIT tokenizer
bash scripts/3_train.sh beauty 42      # 3. generator training and test evaluation
```

Datasets: `beauty`, `sports`, `toys`. Test results (Recall@5/10, NDCG@5/10) are written to
`outputs/<dataset>/s<seed>/generator/metrics_test.json`.

The preprocessed Amazon data follow GRID (Ju et al., 2025).
