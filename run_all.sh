#!/usr/bin/env bash
set -euo pipefail

OFFLINE=0
if [[ "${1:-}" == "--offline" ]]; then
  OFFLINE=1
fi

echo "=== Running tests ==="
python -m pytest -q tests || true

if [[ $OFFLINE -eq 0 ]]; then
  echo "=== Collecting external data (ChEMBL / PubChem) ==="
  python -m src.collect_data \
      --out data/external/vacv_external_raw.csv \
      --doc-timeout 300 \
      --max-activities 50000 || echo "Collection finished with warnings (see skipped_docs.csv)"

  echo "=== Validating collected data ==="
  python -m src.validate_data \
      --in data/external/vacv_external_raw.csv \
      --out data/external/vacv_external_clean.csv || true
else
  echo "=== Offline mode: skipping network collection ==="
fi

echo "=== Training final model (v2) ==="
python -m src.train_v2 \
    --out outputs_v2 \
    --external data/external/vacv_external_clean.csv

echo ""
echo "Done. Main submission file:"
echo "  outputs_v2/submission.csv"
echo ""
echo "Other useful artefacts:"
echo "  outputs_v2/submission_no_analog.csv"
echo "  outputs_v2/submission_analog_strong.csv"
echo "  outputs_v2/analog_changes.csv"
echo "  outputs_v2/cv_report_v2.json"
