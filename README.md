# Constraint-Aware Differentially Private Data Generation

This repository contains the code for **constraint-aware differentially private synthetic data generation**.  
It provides implementations of our method (PrivCI), Prefair baselines (greedy and exponential), and MST-based mechanisms for enforcing constraints under differential privacy.

---

## 📂 Project Structure

- `utils/` – contains utility scripts for defining constraints, preprocessing, and evaluation.
- `evaluations/` - contains the downstream evaluation script.
- `data/` – place to store source datasets (Adult, COMPAS, Dutch, etc.).
- `Prefair/` – implementation of Prefair Greedy and Exponential methods.
- `private-pgm/` – implementation of MST and PrivCI methods.
- `evaluate.py` – evaluation pipeline for machine learning tasks.
- `requirements.txt` – required Python dependencies.

---

## ⚙️ Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/gnamiro/CI-Aware-DP-synth.git
   cd CI_DP_synth
   ```

2. (Recommended) Create and activate a virtual environment:

    On Linux / MacOS:

        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

    On Windows (PowerShell):

        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🛠️ Usage Instructions

### Step 1 – Define Constraints

- Open utils/constraint.py.

- Define your constraints based on the provided examples.

- These constraints will be used in synthetic data generation.

### Step 2 – Prepare the Data

- Place the source dataset (Adult, COMPAS, Dutch, etc.) inside the `data/` directory.

- You can check existing examples in `data/`.

⚡ Quick Run Option

You can either run the provided script `generate_dp.sh` (make sure to have Git Bash or WSL installed on Windows) to automatically create the necessary folders and execute all methods (`mst`, `prefair`, and our method) without manually running each one separately.  

However, if you prefer to change parameters or run specific steps independently, you can follow the instructions below.

---

### Step 3 – Preprocess Data

- Open `utils/preprocess.py`.

- Define your preprocessing steps (e.g., cleaning, encoding, normalization).

- Use the `split_data` method to create 5-fold cross-validation splits on the preprocessed data, while respecting the constraints defined in `utils/constraint.py`.

- Run:

    ```bash
    python utils/preprocess.py --dataset Adult
    ```

### Step 4 – Generate Synthetic Data

You can choose between different methods:

#### Prefair

- Run:

    ```bash
    python Prefair/src/generate_dp.py
    ```

- To switch between greedy and exponential, change the `type` parameter inside `Prefair/src/generate_dp.py`.

#### Private-PGM

- Run MST:

    ```bash
    python private-pgm/mst_generate.py
    ```

- Run PrivCI or Hard Constraint MST:
    ```bash
    python private-pgm/generate_dp.py
    ```

💡 Make sure to set the privacy parameters (`eps`, `delta`) inside these scripts.
We fix `delta = 1e-9` (commonly used in literature).

Synthetic datasets will be generated inside:

```bash
data/{DATASET_NAME}/folds/
```

## 📊 Statistical Tests and Evaluation

1. Conditional Independence Tests

Run:

```bash
python utils/CI_test.py --test_type CMI --eps 10
```

Options:

`CMI`

`Chi`

2. Statistical Distance (Wasserstein, TVD, etc.)

Run:

```bash
python utils/wasserstein_dist.py --dist_type w --eps 10
```

3. End-to-End Evaluation

```bash
python evaluate.py --eps 10 --method MLP --with-privacy
```

This evaluates synthetic datasets using machine learning models under different privacy budgets.

⚠️ Make sure that you have results for all methods (`mst`, `greedy`, `opt`, `privci`, `mst_hard`).
If some methods are missing, comment out the corresponding `load_dataset` calls in evaluation scripts.

⚠️ If you want to use floating-point eps values (e.g., eps=0.1, eps=0.01), update the argument type in the following files:

- `utils/CI_test.py`

- `utils/wasserstein_dist.py`

- `evaluate.py`

Change:

```bash
    parser.add_argument('--eps', type=int, default=None, help='Epsilon value')
```

to:

```bash
    parser.add_argument('--eps', type=float, default=None, help='Epsilon value')
```

