# Variation of Private-PGM: Adding CI Regularization

This repository is a variation of the original [private-pgm](https://github.com/ryan112358/private-pgm) implementation.  
Our main goal is to **extend marginal-based inference with a Conditional Independence (CI) regularizer** in the optimization process, to better enforce fairness and remove spurious correlations while maintaining high data utility.

In addition to the original features of private-pgm, we include scripts and experiments for:

- **CI Tests**  
- **Statistical Distance (e.g., Wasserstein)**  
- **Data Cleaning Evaluation**  

For data generation, please refer to:

- `generate_dp.py`  
- `mst_generation.py`  

---

# Marginal-based estimation and inference for differential privacy

<img src="pgm-logo.png" alt="drawing" width="123"/>

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5548533.svg)](https://doi.org/10.5281/zenodo.5548533)
[![Continuous integration](https://github.com/ryan112358/private-pgm/actions/workflows/main.yml/badge.svg)](https://github.com/ryan112358/private-pgm/actions/workflows/main.yml)

---

## Summary

This is a general-purpose library for estimating discrete distributions from noisy
observations of their marginals. We provide scalable algorithms for solving convex optimization problems over the marginal polytope. From that, we obtain an undirected graphical model (Markov random field) which can be used to:

1. Obtain more accurate estimates of the observed noisy marginals,  
2. Estimate answers to new queries (e.g., marginals) based on a maximum entropy assumption,  
3. Generate synthetic data that approximately preserves those marginals.  

The library is designed with the following core principles in mind:

* **Consistency** – Produces consistent estimates of measured and unmeasured queries.  
* **Utility** – Maximizes use of noisy observations for strong practical performance.  
* **Scalability** – Scales effectively to high-dimensional datasets.  
* **Flexibility** – Supports generic loss functions over marginals.  
* **Extensibility** – Designed to be built upon by future research.  
* **Simplicity** – Easy-to-use APIs, toy examples require only a few lines of code.  

---

## Additional Experiments in This Repository

### 1. Conditional Independence (CI) Test
Run the CI test with:

```bash
python Experiments/CI_test.py --test_type CMI --eps 10
```

- `--test_type`: Type of CI test (`CMI`).  
- `--eps`: Privacy budget parameter.  

---

### 2. Statistical Distance (Wasserstein)
Run Wasserstein distance experiments with:

```bash
python Experiments/wasserstein_dist.py --dist_type w --eps 10
```

- `--dist_type`: Type of distance metric (`w` = Wasserstein).  
- `--eps`: Privacy budget parameter.  

---

### 3. Data Cleaning
Check cleaning results in:

- **`Experiments/data_cleaning.ipynb`**

---

## Notes
- Downstream tasks for **fairness evaluation** will be included soon.  

---

## Installation

### Automatic Setup
```bash
pip install git+https://github.com/ryan112358/private-pgm.git
```

### Manual Setup
We require Python >= 3.9. Install dependencies with:

```bash
pip install -r requirements.txt
```

Add the `src` folder to the `PYTHONPATH`. On Ubuntu, add to your `.bashrc`:

```bash
PYTHONPATH=$PYTHONPATH:/path/to/private-pgm/src
```

Check installation by running tests:

```bash
cd /path/to/private-pgm/test
pytest
```

---

## Codebase Organization
- Core library: `src/mbi`  
- Example usage: `examples/`  
- Mechanisms: `mechanisms/`  
- Our experiments: `Experiments/`  

For data generation:  
- **`generate_dp.py`**  
- **`mst_generation.py`**  

---

## Contributing
Contributions are welcome, including:
- New functionality (estimators, generators, etc.)  
- Bug fixes and performance improvements  
- Documentation updates  
- Extensions of mechanisms  

---

## References
- McKenna et al. (2019, 2021, 2023). See main repository for detailed papers.
