# Setup Guide

## Prerequisites

- Python 3.10 or higher
- conda (recommended) or pip

## Option A: conda (recommended)

```bash
conda create -n bayes-astro python=3.11
conda activate bayes-astro
conda install -c conda-forge numpy scipy matplotlib jupyter
pip install -r requirements.txt
```

## Option B: pip + virtualenv

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
.venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

## Launch Notebooks

```bash
jupyter lab
```

Navigate to `chapters/ch01_probability_foundations/notebook.ipynb` to begin.

## Test Your Installation

```python
import numpy as np
import emcee
import dynesty
import batman
import lightkurve as lk
import celerite2
print("All dependencies installed successfully!")
```

## GitHub Setup (for your own fork)

```bash
git clone https://github.com/YOUR_USERNAME/bayesian-astrophysics.git
cd bayesian-astrophysics
git remote add upstream https://github.com/ORIGINAL/bayesian-astrophysics.git
```
