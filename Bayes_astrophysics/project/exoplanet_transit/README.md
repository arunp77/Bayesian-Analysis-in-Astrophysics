# 🪐 Capstone Project: Exoplanet Transit Light Curve Analysis

> **Applying all 8 chapters to a real astrophysical inference problem**

---

## Project Overview

We will perform a complete Bayesian analysis of an exoplanet transit — fitting the physical transit model to photometric light curve data to infer the planet's radius, orbital period, inclination, and system parameters.

This project demonstrates every technique in the curriculum:
- Chapter 1-3: Noise model construction, likelihood, priors
- Chapter 4: MCMC parameter estimation with emcee
- Chapter 5: Nested sampling with dynesty for model comparison
- Chapter 6: Comparing transit vs. no-transit vs. EB models
- Chapter 7: GP noise model for stellar variability
- Chapter 8: Hierarchical analysis across multiple transits

---

## The Physics: Transit Geometry

When an exoplanet transits its host star, it blocks a fraction of starlight:

$$\frac{\Delta F}{F} \approx \left(\frac{R_p}{R_*}\right)^2 = k^2$$

The transit duration:
$$T_{14} = \frac{P}{\pi} \arcsin\!\left(\frac{R_*}{a}\sqrt{(1+k)^2 - b^2}\right)$$

where:
- $k = R_p/R_*$ — planet-to-star radius ratio
- $b = a\cos i / R_*$ — impact parameter
- $a$ — semi-major axis (from Kepler's third law: $a^3 = G M_* P^2 / 4\pi^2$)
- $i$ — orbital inclination
- $P$ — orbital period

---

## The Model: Mandel-Agol (2002)

The transit light curve model with quadratic limb darkening:

$$F(t) = F_0 \cdot \lambda(z(t),\, k,\, u_1,\, u_2)$$

where $z(t)$ is the projected sky-plane separation (in units of $R_*$) and $\lambda$ is the Mandel-Agol flux ratio.

The limb-darkened stellar intensity profile:
$$I(\mu) = 1 - u_1(1-\mu) - u_2(1-\mu)^2, \quad \mu = \cos\theta$$

---

## Parameter Space

### Physical Parameters (θ):
| Parameter | Symbol | Prior | Physical Meaning |
|-----------|--------|-------|-----------------|
| Period | P | Log-Uniform(0.5d, 30d) | Orbital period |
| Time of transit center | t₀ | Uniform(t_start, t_end) | Reference transit time |
| Radius ratio | k = Rp/R★ | Uniform(0.01, 0.3) | Planet size relative to star |
| Impact parameter | b | Uniform(0, 1+k) | Transit chord position |
| Stellar density | ρ★ | Gaussian from spectroscopy | Sets transit shape |
| Limb darkening | u₁, u₂ | Informative from Claret tables | Stellar atmosphere |
| Baseline flux | F₀ | Gaussian(1, 0.001) | Out-of-transit flux |

### GP Noise Parameters (φ):
| Parameter | Symbol | Prior | Meaning |
|-----------|--------|-------|---------|
| GP amplitude | σ_GP | Log-Uniform(10⁻⁵, 10⁻¹) | Stellar variability amplitude |
| GP timescale | ℓ_GP | Log-Uniform(0.01d, 10d) | Correlation timescale |
| White noise | σ_w | Log-Uniform(10⁻⁵, 10⁻²) | Jitter / uncorrelated noise |

---

## Analysis Pipeline

```
Phase 1: Data Preparation
├── Load light curve (TESS/Kepler FITS)
├── Sigma-clip outliers
├── Normalise and detrend long-term trends
└── Visualise raw data

Phase 2: Transit Timing (Box Least Squares)
├── BLS periodogram for period finding
├── Phase-fold on best period
└── Initial parameter estimates

Phase 3: MCMC Fit (emcee)
├── Define log_posterior = log_likelihood + log_prior
├── Burn-in (500 steps, 64 walkers)
├── Production (5000 steps)
├── Convergence: R̂ < 1.01, N_eff > 500
└── Corner plot of posteriors

Phase 4: GP+Transit Joint Fit
├── Add GP covariance to likelihood
├── Joint sampling of transit + GP hyperparameters
└── Compare residuals before/after GP

Phase 5: Model Comparison (dynesty)
├── Model M0: Flat baseline (no transit)
├── Model M1: Transit only
├── Model M2: Transit + GP
├── Model M3: Eclipsing binary
└── Compute ln B_ij for all pairs

Phase 6: Physical Results
├── Compute Rp from k and known R★
├── Compute a from ρ★ and P (Kepler III)
├── Derive equilibrium temperature: T_eq
└── Compare to published values
```

---

## Expected Outputs

1. **Corner plot** showing posterior correlations between (k, b, t₀, P)
2. **Phase-folded light curve** with model and 1σ/2σ uncertainty bands
3. **Residuals** before and after GP detrending
4. **Evidence table** comparing all four models
5. **Physical parameter summary** table with 16th/50th/84th percentiles

---

## Dataset

We will use **TESS** data for **WASP-39b** — a well-characterised hot Jupiter with:
- P = 4.055 days
- k ≈ 0.14 (Rp ≈ 1.27 R_Jup)
- b ≈ 0.43

Data available via `lightkurve`:
```python
import lightkurve as lk
result = lk.search_lightcurve("WASP-39", mission="TESS")
lc = result[0].download().normalize().remove_outliers()
```

---

## Dependencies

```
batman-package    # Mandel-Agol transit model
emcee             # MCMC sampling
dynesty           # Nested sampling
celerite2         # Fast GP for time series
lightkurve        # TESS/Kepler data access
corner            # Posterior corner plots
astropy           # Astronomy utilities
numpy, scipy      # Numerics
matplotlib        # Visualisation
```

Install:
```bash
pip install batman-package emcee dynesty celerite2 lightkurve corner astropy
```

---

## Timeline

| Week | Focus |
|------|-------|
| 1 | Data download, exploration, BLS period finding |
| 2 | Build transit model, run initial MCMC |
| 3 | Add GP noise model, joint fit |
| 4 | Nested sampling, model comparison, write-up |

---

## 🔗 Navigation

[← Chapter 8: Hierarchical Models](../../chapters/ch08_hierarchical_models/README.md)  
[→ Project Code](./analysis.py)
