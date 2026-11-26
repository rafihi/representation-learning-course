#!/usr/bin/env bash
# End-to-end smoke test of every Part A task using the course lec3/train.py entry point.
# Synthetic data, one epoch / tiny subsets, no dataset downloads.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
COURSE_TRAIN="../lec3/train.py"
if [[ ! -f "$COURSE_TRAIN" ]]; then
  echo "Course train.py not found: $COURSE_TRAIN" >&2
  exit 2
fi

export SSL_FAKE_DATA=1
rm -rf runs_smoke results_smoke
mkdir -p /tmp/midterm_a_smoke_cfgs /tmp/midterm_a_smoke_logs

run() {
  local config_name="$1"
  local config_copy="/tmp/midterm_a_smoke_cfgs/${config_name}"
  local log_file="/tmp/midterm_a_smoke_logs/${config_name%.yaml}.log"
  sed 's#runs/#runs_smoke/#g; s#results/#results_smoke/#g' "configs/${config_name}" > "$config_copy"
  echo "=== ${config_name} via ${COURSE_TRAIN} ==="
  if ! python "$COURSE_TRAIN" --config "$config_copy" --smoke --set trainer.progress=false >"$log_file" 2>&1; then
    echo "FAILED: ${config_name}. Last 80 log lines:" >&2
    tail -n 80 "$log_file" >&2
    return 1
  fi
  tail -n 8 "$log_file"
}

for c in a2_dae_final.yaml a3_vanilla_ae_final.yaml a1_tune_dae.yaml a4_eval.yaml a5_finetune.yaml a6_robustness.yaml a7_interpret.yaml; do
  run "$c"
done

echo "ALL PART-A SMOKE TESTS PASSED. Figures:"
find results_smoke -name '*.png' -type f | sort
