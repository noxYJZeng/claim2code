#!/usr/bin/env bash
# ============================================================
# Claim2Code — run the full evaluation on this machine
#
#   scripts/eval-local.sh            preflight, then every evaluator + report
#   scripts/eval-local.sh --resume   continue an interrupted verification run
#   scripts/eval-local.sh --check    preflight only
#
# Uses the models and database configured in .env (for example LM Studio
# on localhost). Verification takes about two hours with a local 27B
# model; on macOS the run holds off sleep with caffeinate. The log goes
# to evaluation/logs/ (gitignored).
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

MODE=run
EVAL_ARGS=()
case "${1:-}" in
    "") ;;
    --resume) EVAL_ARGS+=(--resume) ;;
    --check) MODE=check ;;
    *) echo "usage: $0 [--resume | --check]" >&2; exit 2 ;;
esac

# The API's interpreter: $PYTHON, else a project venv, else python3.
if [ -z "${PYTHON:-}" ]; then
    for candidate in "$ROOT/.venv/bin/python" "$HOME/.venvs/claim2code/bin/python"; do
        if [ -x "$candidate" ]; then PYTHON="$candidate"; break; fi
    done
fi
PYTHON="${PYTHON:-python3}"
if ! "$PYTHON" -c "import claim2code_api" 2>/dev/null \
    && ! PYTHONPATH="$ROOT/apps/api" "$PYTHON" -c "import claim2code_api" 2>/dev/null; then
    echo "✗ $PYTHON cannot import the API package; install it (pip install -e apps/api) or set PYTHON=" >&2
    exit 1
fi
export PYTHONPATH="$ROOT/apps/api:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

[ -f "$ROOT/.env" ] || { echo "✗ .env not found; copy .env.example and set the model and database" >&2; exit 1; }

"$PYTHON" evaluation/preflight.py || {
    echo "" >&2
    echo "Fix the ✗ items above and run again." >&2
    exit 1
}
[ "$MODE" = check ] && exit 0

mkdir -p evaluation/logs
LOG="evaluation/logs/eval-$(date -u +%Y%m%dT%H%M%SZ).log"
echo ""
echo "Running every evaluator; log: $LOG"
KEEP_AWAKE=()
command -v caffeinate >/dev/null && KEEP_AWAKE=(caffeinate -i)

# ${a[@]+"${a[@]}"}: macOS bash 3.2 treats an empty array as unset under set -u.
set +e
${KEEP_AWAKE[@]+"${KEEP_AWAKE[@]}"} "$PYTHON" evaluation/evaluate_all.py ${EVAL_ARGS[@]+"${EVAL_ARGS[@]}"} 2>&1 \
    | tee "$LOG" | grep --line-buffered -E "expected=|ERROR|Traceback|wrote "
status=${PIPESTATUS[0]}
set -e

echo ""
"$PYTHON" - <<'EOF'
import json
from pathlib import Path

path = Path("evaluation/results/verification.json")
if path.exists():
    data = json.loads(path.read_text())
    agg, prov = data["aggregate"], data.get("provenance") or {}
    code, models = prov.get("code", {}), prov.get("models", {})
    print(f"Verification at {str(code.get('git_commit'))[:7]}{' (dirty)' if code.get('dirty') else ''}"
          f" with {models.get('chat_model')}:")
    base = agg.get("majority_class_baseline") or {}
    print(f"  accuracy {agg['status_accuracy']:.1%}  (always-{base.get('label')} baseline"
          f" {base.get('status_accuracy', float('nan')):.1%})")
    print(f"  support {agg['support_rate']:.1%}  LLM errors {agg['n_llm_errors']}")
EOF
echo "Report: evaluation/report.md"
[ "$status" -eq 0 ] || echo "✗ an evaluator exited with status $status; see $LOG" >&2
exit "$status"
