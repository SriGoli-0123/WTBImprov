#!/usr/bin/env bash
#
# Run the whole benchmark -- all 256 sessions -- twice: once without PRISM
# and once with it, then compare the two turn by turn.
#
#   bash method/run_full.sh                       # both arms, default model
#   bash method/run_full.sh --arm prism           # just the current method arm
#   bash method/run_full.sh --arm concord         # retained CONCORD ablation
#   bash method/run_full.sh --arm gavel           # retained GAVEL-v2 ablation
#   bash method/run_full.sh --arm baseline        # just the plain arm
#   bash method/run_full.sh --arm grounded        # legacy grounded wrapper
#   bash method/run_full.sh --tag v2              # keep it separate from an earlier run
#
# Safe to re-run. Sessions that already have a result are skipped, so if the
# run dies halfway through, the same command picks up where it stopped. Pass
# --fresh to throw the old results away and start over.
#
# Run from WildToolBench/wild-tool-bench, with the model server already up in
# another terminal.

set -euo pipefail

MODEL="Qwen/Qwen2.5-7B-Instruct"
THREADS=8
ARM="both"
TAG="full"
FRESH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)   MODEL="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --arm)     ARM="$2"; shift 2 ;;
        --tag)     TAG="$2"; shift 2 ;;
        --fresh)   FRESH="--allow-overwrite"; shift ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "unknown option: $1"; exit 1 ;;
    esac
done

MODEL_DIR="${MODEL//\//_}"
LOGS="logs_${TAG}"
mkdir -p "$LOGS"

# ---------------------------------------------------------------- preflight
echo "checking the model server..."
BASE_URL=$(grep -E '^OPENAI_BASE_URL' .env 2>/dev/null | cut -d= -f2- | tr -d '"' || true)
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
if ! curl -sf "${BASE_URL}/models" >/dev/null; then
    echo "  cannot reach ${BASE_URL} -- is the vLLM server running?"
    exit 1
fi
echo "  ok: ${BASE_URL}"

echo "checking which handler each arm picks up..."
python3 -c "
from wtb.model_handler.handler_map import HANDLER_MAP as H
print('  baseline ->', H['${MODEL}'].__name__)
"
WTB_METHOD=prism python3 -c "
from wtb.model_handler.handler_map import HANDLER_MAP as H
name = H['${MODEL}'].__name__
print('  prism    ->', name)
raise SystemExit(0 if name == 'PrismHandler' else 'the PRISM arm is not wired up -- stopping')
"

# ------------------------------------------------------------------- arms
# No --run-ids, so every session in the file is used: all 256.
run_arm () {
    local name="$1" env_prefix="$2"
    local rdir="result_${TAG}_${name}" sdir="score_${TAG}_${name}"

    echo
    echo "=== ${name} arm -> ${rdir} ==="
    date

    env ${env_prefix} python3 -u -m wtb.openfunctions_evaluation \
        --model "${MODEL}" \
        --num-threads "${THREADS}" \
        --result-dir "${rdir}" \
        ${FRESH} 2>&1 | tee "${LOGS}/${name}_generate.log"

    python3 -u -m wtb.eval_runner \
        --model "${MODEL_DIR}" \
        --result-dir "${rdir}" \
        --score-dir "${sdir}" 2>&1 | tee "${LOGS}/${name}_score.log"

    echo "--- ${name} headline ---"
    cat "${sdir}/${MODEL_DIR}"/*metric*.json 2>/dev/null || true
    date
}

if [[ "$ARM" == "both" || "$ARM" == "baseline" ]]; then
    run_arm baseline ""
fi
if [[ "$ARM" == "both" || "$ARM" == "prism" ]]; then
    run_arm prism "WTB_METHOD=prism"
fi
if [[ "$ARM" == "concord" ]]; then
    run_arm concord "WTB_METHOD=concord"
fi
if [[ "$ARM" == "gavel" ]]; then
    run_arm gavel "WTB_METHOD=gavel"
fi
if [[ "$ARM" == "grounded" ]]; then
    run_arm grounded "WTB_METHOD=grounded"
fi

# ---------------------------------------------------------------- compare
if [[ "$ARM" == "both" ]]; then
    echo
    echo "=== fixed / broke, turn by turn ==="
    python3 method/compare.py \
        "score_${TAG}_baseline/${MODEL_DIR}" \
        "score_${TAG}_prism/${MODEL_DIR}" | tee "${LOGS}/compare.txt"
fi

echo
echo "done. logs in ${LOGS}/"
