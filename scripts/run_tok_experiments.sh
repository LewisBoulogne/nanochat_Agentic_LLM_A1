#!/usr/bin/env bash
# Task 1: train and evaluate BPE tokenizers at two vocabulary sizes.
# Usage: bash scripts/run_tok_experiments.sh
set -euo pipefail

: "${NANOCHAT_BASE_DIR:?set NANOCHAT_BASE_DIR first}"
mkdir -p logs

python -m nanochat.dataset -n 6

for V in 8192 32768; do
  python -m scripts.tok_train --vocab-size="$V" --max-chars=500000000 --doc-cap=10000 \
    2>&1 | tee "logs/tok_train_${V}.log"
  python -m scripts.tok_eval 2>&1 | tee "logs/tok_eval_${V}.log"
  rm -rf "$NANOCHAT_BASE_DIR/tokenizer_v${V}"
  cp -r "$NANOCHAT_BASE_DIR/tokenizer" "$NANOCHAT_BASE_DIR/tokenizer_v${V}"
done

# leave the 32k tokenizer active for pretraining
rm -rf "$NANOCHAT_BASE_DIR/tokenizer"
cp -r "$NANOCHAT_BASE_DIR/tokenizer_v32768" "$NANOCHAT_BASE_DIR/tokenizer"

python scripts/tok_compare.py 2>&1 | tee logs/tok_compare.log
python scripts/toy_bpe.py --merges 40 2>&1 | tee logs/toy_bpe.log
