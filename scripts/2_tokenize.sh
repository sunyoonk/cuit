#!/bin/bash
# Step 2. CF labels and the CUIT tokenizer.
# usage: bash scripts/2_tokenize.sh [dataset=beauty] [seed=42]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
DS=${1:-beauty}; SEED=${2:-42}
$PY -m cuit.cf.labels dataset=$DS
$PY -m cuit.tokenizer.run dataset=$DS seed=$SEED
