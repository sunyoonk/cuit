#!/bin/bash
# Step 1. Data: preprocessed Amazon 2014 (Beauty, Sports, Toys; P5 leave-last-out split) and item text embeddings.
# usage: bash scripts/1_data.sh [datasets, default: beauty sports toys]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
mkdir -p data
if [ ! -d data/amazon/beauty ]; then
  $PY -m gdown 1B5_q_MT3GYxmHLrMK0-lAqgpbAuikKEz -O data/amazon_data.zip
  unzip -q data/amazon_data.zip -d data -x '__MACOSX/*'
  mv data/amazon_data data/amazon
  find data/amazon -name .DS_Store -delete
  rm data/amazon_data.zip
fi
for DS in ${*:-beauty sports toys}; do
  $PY -m cuit.data.text_embed dataset=$DS
done
