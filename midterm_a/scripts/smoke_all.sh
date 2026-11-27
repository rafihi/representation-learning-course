#!/usr/bin/env bash
# End-to-end smoke test of every Part A task (validation AND final-test configs) through train.py.
# Synthetic data, one epoch / tiny subsets, no dataset downloads. Writes only to runs_smoke/ and results_smoke/.
#   SSL_TINY_BACKBONE=1 bash scripts/smoke_all.sh      # CPU (tiny CNN instead of ResNet-18)
#   bash scripts/smoke_all.sh                          # GPU, real ResNet-18
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export SSL_FAKE_DATA=1
TMP_CFG="$(mktemp -d)"
LOGS="$(mktemp -d)"
rm -rf runs_smoke results_smoke

# Copy every config, redirecting outputs (runs/ -> runs_smoke/, results/ -> results_smoke/) and `extends:` to the copies.
for f in configs/*.yaml; do
  sed "s#runs/#runs_smoke/#g; s#results/#results_smoke/#g; s#extends: configs/#extends: ${TMP_CFG}/#g" "$f" > "${TMP_CFG}/$(basename "$f")"
done
sed -i "s#base_config: configs/#base_config: ${TMP_CFG}/#g" "${TMP_CFG}/a1_tune_dae.yaml"

run() {
  local name="$1"
  echo "=== ${name} ==="
  if ! python train.py --config "${TMP_CFG}/${name}" --smoke --set trainer.progress=false >"${LOGS}/${name%.yaml}.log" 2>&1; then
    echo "FAILED: ${name}. Last 80 log lines:" >&2
    tail -n 80 "${LOGS}/${name%.yaml}.log" >&2
    return 1
  fi
  tail -n 4 "${LOGS}/${name%.yaml}.log"
}

for c in a1_tune_dae.yaml a2_dae_60ep.yaml a3_vanilla_ae_60ep.yaml a2_dae_final.yaml a3_vanilla_ae_final.yaml a2_dynamics.yaml \
         a4_eval.yaml a5_finetune.yaml a6_robustness.yaml a7_interpret.yaml \
         a4_eval_test.yaml a5_finetune_test.yaml a6_robustness_test.yaml a7_interpret_test.yaml; do
  run "$c"
done

echo "ALL PART-A SMOKE TESTS PASSED. Figures:"
find results_smoke -name '*.png' -type f | sort
