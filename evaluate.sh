#!/bin/bash
set -e  # stop if any command fails

echo "[STEP 1] Running with CMI evaluation"
python utils/CI_test.py --test_type CMI

echo "[STEP 2] Running with Chi evaluation"
python utils/CI_test.py --test_type Chi

echo "[STEP 3] Running with w_dist evaluation"
python utils/wasserstein_dist.py --dist_type w

echo "[STEP 4] Running with TVD evaluation"
python utils/wasserstein_dist.py --dist_type tvd

echo "[STEP 5] Running with KL evaluation"
python utils/wasserstein_dist.py --dist_type kl

echo "[STEP 6] Running with AUC on LR evaluation"
python evaluate.py --with-private --method LR

echo "[STEP 7] Running with AUC on MLP evaluation"
python evaluate.py --with-private --method MLP


echo "[DONE] All steps finished successfully."
