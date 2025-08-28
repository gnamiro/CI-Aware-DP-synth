#!/bin/bash
set -e  # stop if any command fails

echo "[STEP 1] Preprocessing data..."
python utils/preprocess.py

echo "[STEP 2] Running Prefair Greedy/Exponential method..."
python Prefair/src/generate_dp.py

echo "[STEP 3] Running MST method..."
python private-pgm/mst_generate.py

echo "[STEP 4] Running PrivCI / Hard Constraint MST..."
python private-pgm/generate_dp.py

echo "[DONE] All steps finished successfully."
