# 🌍 Bayesian Statistics for Atmospheric Remote Sensing

> A targeted curriculum bridging Bayesian inference theory with the physical
> measurement problems in atmospheric science and remote sensing.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-brightgreen.svg)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-notebooks-orange.svg)](https://jupyter.org)

---

## Why Bayesian Methods in Atmospheric Remote Sensing?

Atmospheric remote sensing is fundamentally an **inverse problem**: we measure
upwelling or downwelling electromagnetic radiation, and ask what state of the
atmosphere produced it. This is hard for several reasons:

- The measurement is indirect — we never "see" temperature or aerosol loading directly
- The problem is **ill-posed** — multiple atmospheric states can produce the same spectrum
- Observations are noisy (instrument noise, calibration uncertainty)
- Forward models (radiative transfer) are approximate
- Prior knowledge (climatology, model fields) is genuinely useful

The Bayesian framework handles all of these naturally. The posterior:

    P(x | y) ∝ P(y | x) · P(x)
    
    x = atmospheric state (temperature, gas mixing ratios, aerosols...)
    y = measured radiances / reflectances / brightness temperatures
    P(y|x) = radiative transfer forward model + instrument noise
    P(x) = prior from climatology / NWP model

---

## Curriculum Structure

| Chapter | Title | Core Method | RS Application |
|---------|-------|-------------|----------------|
| 01 | Radiative Transfer as a Forward Model | Beer-Lambert, Planck, RTE | The physics behind the measurement |
| 02 | The Measurement Equation | Jacobians, Weighting Functions | How atmosphere maps to radiance |
| 03 | Optimal Estimation (OE) | Gaussian analytical inversion | MERRA-2, ECMWF retrievals |
| 04 | MCMC Retrievals | emcee, convergence | Non-Gaussian posteriors |
| 05 | Gaussian Processes in Remote Sensing | Spatiotemporal GP | Kriging, spatial interpolation |
| 06 | Hierarchical Models for Ensembles | Multi-layer priors | Ensemble averaging, bias correction |

---

## Capstone Project

**Aerosol Optical Depth Retrieval from Synthetic MODIS-like Observations**

Using a simplified forward model, we retrieve:
- Aerosol Optical Depth (AOD) at 550 nm
- Ångström exponent (spectral dependence)
- Surface reflectance

with full Bayesian uncertainty quantification, using both Optimal Estimation
and MCMC, and comparing their posteriors.

---

## Quick Start

```bash
pip install numpy scipy matplotlib jupyter emcee corner
jupyter lab chapters/ch01_radiative_transfer/notebook.ipynb
```

---

## Key References

- Rodgers (2000): *Inverse Methods for Atmospheric Sounding*  ← The bible of OE
- Tarantola (2005): *Inverse Problem Theory*
- Dee et al. (2011): ECMWF ERA-Interim (Bayesian data assimilation)
- Levy et al. (2013): MODIS aerosol retrieval algorithm
- Lary et al. (2016): ML/Bayes in atmospheric science review
