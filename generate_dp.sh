#!/bin/bash
set -e  # stop if any command fails

echo "[STEP 1] Preprocessing data..."
python utils/preprocess.py --dataset Compas

echo "[STEP 2] Running Prefair Greedy method..."
python Prefair/src/generate_dp.py --method G

echo "[STEP 2.5] Running Prefair Exponential method..."
python Prefair/src/generate_dp.py --method E

echo "[STEP 3] Running MST method..."
python private-pgm/mst_generate.py

echo "[STEP 4] Running Hard Constraint MST..."
python private-pgm/generate_dp.py --method HC
echo "[STEP 4.5] Running Hard Constraint MST..."
python private-pgm/generate_dp.py --method PrivCI


echo "[DONE] All steps finished successfully."
