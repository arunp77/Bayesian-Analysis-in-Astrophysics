# 🌌 Bayesian Analysis in Astrophysics — From Scratch

> A complete, self-contained curriculum for learning Bayesian statistical methods as applied to complex astrophysical data.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-brightgreen.svg)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-notebooks-orange.svg)](https://jupyter.org)

---

## 📖 Overview

This repository is a structured, chapter-by-chapter learning path that takes you from **first principles of probability** through to **cutting-edge Bayesian inference techniques** used in real astrophysics research — culminating in a fully worked **Exoplanet Transit Light Curve Analysis** project.

The curriculum is designed for:

- Physics/astronomy students with basic calculus and linear algebra
- Engineers transitioning into data science for scientific applications
- Researchers wanting a rigorous foundation before applying black-box tools

---

## 🗂️ Repository Structure

```
bayesian-astrophysics/
│
├── README.md                        ← You are here
├── SETUP.md                         ← Environment setup guide
├── CONTRIBUTING.md                  ← Contribution guidelines
│
├── chapters/                        ← Core curriculum (8 chapters)
│   ├── ch01_probability_foundations/
│   ├── ch02_bayes_theorem/
│   ├── ch03_likelihood_priors/
│   ├── ch04_mcmc/
│   ├── ch05_nested_sampling/
│   ├── ch06_model_comparison/
│   ├── ch07_gaussian_processes/
│   └── ch08_hierarchical_models/
│
├── docs/                            ← Extended theory documents
│   ├── math_reference.md
│   ├── glossary.md
│   └── further_reading.md
│
├── notebooks/                       ← Interactive Jupyter notebooks
├── project/                         ← Capstone: Exoplanet Transit Analysis
│   ├── exoplanet_transit/
│   └── results/
│
├── scripts/                         ← Reusable Python utilities
├── data/                            ← Sample datasets
├── figures/                         ← Generated plots and diagrams
└── tests/                           ← Unit tests for utility functions
```

---

## 📚 Curriculum at a Glance

| Chapter | Title                   | Key Concepts                                                   |
| ------- | ----------------------- | -------------------------------------------------------------- |
| 01      | Probability Foundations | Sample spaces, axioms, conditional probability, distributions  |
| 02      | Bayes' Theorem          | The fundamental theorem, updating beliefs, conjugate priors    |
| 03      | Likelihood & Priors     | Likelihood functions, prior elicitation, posterior derivation  |
| 04      | MCMC Methods            | Metropolis-Hastings, Gibbs sampling, convergence diagnostics   |
| 05      | Nested Sampling         | MultiNest, dynesty, evidence computation                       |
| 06      | Model Comparison        | Bayes factors, AIC/BIC/DIC, Occam's razor                      |
| 07      | Gaussian Processes      | GP regression, covariance kernels, hyperparameter optimization |
| 08      | Hierarchical Models     | Hyperpriors, population inference, multi-level models          |

---

## 🚀 Capstone Project

**[Exoplanet Transit Light Curve Analysis](project/exoplanet_transit/)**

We apply every technique from the curriculum to fit a physical model to real photometric data:

- Transit geometry and the Mandel-Agol light curve model
- Full posterior inference with MCMC (emcee) and nested sampling (dynesty)
- GP noise model for stellar variability
- Hierarchical analysis across multiple transits

---

## ⚙️ Quick Start

```bash
# Clone the repository
git clone https://github.com/arunp77/Bayesian-statistics.git

# Set up the environment
conda env create -f environment.yml
conda activate bayes-astro

# Or with pip
pip install -r requirements.txt

# Launch the first chapter notebook
jupyter notebook chapters/ch01_probability_foundations/notebook.ipynb
```

---

## 📐 Mathematical Prerequisites

This curriculum assumes familiarity with:

- Calculus (integration, differentiation)
- Linear algebra (matrices, eigenvalues)
- Basic Python programming

Everything else — statistics, Bayesian reasoning, astrophysics concepts — is built from first principles.

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

_"The theory of probabilities is at bottom nothing but common sense reduced to calculus."_  
— Pierre-Simon Laplace
