#!/usr/bin/env bash
# End-to-end smoke test of EVERY Part A task on synthetic data (no downloads, a few minutes).
# Outputs go to runs_smoke/ and results_smoke/ (the configs are copied with the paths rewritten).
# On a machine WITHOUT a GPU also export SSL_TINY_BACKBONE=1 (swaps ResNet-18 for a tiny CNN; tests only!).
set -euo pipefail
export SSL_FAKE_DATA=1
rm -rf runs_smoke results_smoke
mkdir -p /tmp/smoke_cfgs
run() {  # run <config file name>
  sed "s#runs/#runs_smoke/#g; s#results/#results_smoke/#g" "configs/$1" > "/tmp/smoke_cfgs/$1"
  echo "=== $1"; python train.py --config "/tmp/smoke_cfgs/$1" --smoke --set trainer.progress=false 2>&1 | grep -E "^\[done\]|Error|Traceback|assert" || true
}
for c in a2_dae_final a3_vanilla_ae_final a1_tune_dae a4_eval a5_finetune a6_robustness a7_interpret; do run $c.yaml; done
echo "SMOKE TEST FINISHED. Figures:"; find results_smoke -name "*.png" | sort
