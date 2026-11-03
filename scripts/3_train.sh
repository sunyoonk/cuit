#!/bin/bash
# Step 3. Generator: training with validation Recall@5 checkpoint selection, then one evaluation on the test set.
# usage: bash scripts/3_train.sh [dataset=beauty] [seed=42]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
DS=${1:-beauty}; SEED=${2:-42}
$PY -m cuit.generator.train dataset=$DS seed=$SEED
