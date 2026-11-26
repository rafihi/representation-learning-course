#!/usr/bin/env bash
# Run the final Part-A Test evaluation through the local Part-A train.py entry point.
# Run from midterm_a/. Needs the checkpoints under runs/ and the validation fine-tuning run (results/a5_finetune/selected/).
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ENTRY_POINT="train.py"
[[ -f "$ENTRY_POINT" ]] || { echo "Missing Part-A entry point: $ENTRY_POINT" >&2; exit 2; }

for cfg in \
  a4_eval_test.yaml \
  a5_finetune_test.yaml \
  a6_robustness_test.yaml \
  a7_interpret_test.yaml
do
  echo "============================================================"
  echo "FINAL TEST: $cfg"
  python "$ENTRY_POINT" --config "configs/$cfg"
done

echo "All Part-A final-test tasks completed. Results: results/final_test/"
