#!/bin/bash
# Run the browser UI/UX suite into a self-contained artifact bundle.
# Usage: scripts/run_ui_tests.sh [--analyze] [extra pytest args...]
# See docs/UI_TEST_PLAYBOOK.md for the full pipeline.
set -u
cd "$(dirname "$0")/.."

ANALYZE=0
if [ "${1:-}" = "--analyze" ]; then ANALYZE=1; shift; fi

RUN_DIR="artifacts/ui-run-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"
export UI_RUN_DIR="$PWD/$RUN_DIR"

.venv/bin/python -m pytest tests/ui -m ui -q \
  --browser chromium \
  --junitxml="$RUN_DIR/junit.xml" \
  --screenshot only-on-failure \
  --tracing retain-on-failure \
  --output "$RUN_DIR/pw" \
  "$@"
TEST_EXIT=$?

echo "run dir: $RUN_DIR"

if [ $ANALYZE -eq 1 ]; then
  .venv/bin/python scripts/analyze_ui_run.py "$RUN_DIR"
  ANALYZE_EXIT=$?
  [ $ANALYZE_EXIT -ne 0 ] && exit $ANALYZE_EXIT
fi

exit $TEST_EXIT
