# Chapter 8: Hierarchical Bayesian Models

> *"The whole is more than the sum of its parts — and Bayesian inference tells you exactly how much more."*

---

## 8.1 The Population Inference Problem

Suppose we observe N objects — exoplanets, supernovae, galaxy clusters — each with noisy data D_i. We want to know:

1. What are the individual parameters theta_i for each object?
2. What is the **population distribution** P(theta | phi) from which all theta_i are drawn?
3. What can the population tell us about cosmology, stellar physics, or planetary formation?

Fitting each object separately:
- Ignores population structure
- Over-fits to noise in sparsely observed objects
- Cannot leverage shared information

Hierarchical Bayesian modelling (HBM) solves all three problems.

---

## 8.2 The Three-Level Hierarchy

    phi ~ P(phi)                    [Hyperprior — top level]
    theta_i ~ P(theta_i | phi)      [Population model — middle level]  
    D_i ~ P(D_i | theta_i)          [Likelihood — data level]

**Joint posterior:**

    P(phi, {theta_i} | {D_i}) ∝ P(phi) * prod_i [ P(D_i | theta_i) * P(theta_i | phi) ]

This is the full generative model. Everything flows from it.

---

## 8.3 Marginalising Individual Parameters

If only the population parameters phi are of interest:

    P(phi | {D_i}) ∝ P(phi) * prod_i [ integral P(D_i | theta_i) * P(theta_i | phi) d theta_i ]

Each integral is the **marginal likelihood** for object i — the evidence for that object under the population model phi.

This integral is:
- Analytically tractable for conjugate models (e.g. Gaussian-Gaussian)
- Approximated by importance sampling from individual posteriors
- Sampled jointly in modern PPL frameworks (PyMC, NumPyro, Stan)

---

## 8.4 Shrinkage: Borrowing Strength

The defining property of hierarchical models: **partial pooling**.

Individual estimates are shrunk toward the population mean. The amount of shrinkage depends on:
- How noisy the individual measurement is
- How tight the population distribution is

For a Gaussian population N(mu, tau^2) with Gaussian likelihoods N(theta_i, sigma_i^2):

    E[theta_i | D_i, mu, tau] = (1 - B_i) * y_i + B_i * mu

where B_i = sigma_i^2 / (sigma_i^2 + tau^2) is the **shrinkage factor**.

- B_i → 1 (heavy shrinkage) when measurement noise sigma_i >> population spread tau
- B_i → 0 (no shrinkage) when measurement is much more precise than the population spread

**Stein's paradox (1956):** For d ≥ 3 parameters, shrinkage estimators have strictly lower total MSE than the MLE, even when the parameters are unrelated. Hierarchical models achieve this automatically.

---

## 8.5 PyMC Implementation: Stellar Age-Activity Relation

**Science question:** Stars spin down as they age (magnetic braking). We want to infer the population-level age-rotation relation from N=50 stars with noisy age estimates from isochrone fitting and rotation periods from light curves.

**Model:**
- log P_rot = alpha * log(age) + beta + N(0, sigma_int)   [power law + scatter]
- age_i, P_rot,i are measured with Gaussian noise

```python
import pymc as pm
import numpy as np

with pm.Model() as age_rotation_model:

    # ── Hyperpriors (population-level) ──────────────────
    alpha     = pm.Normal("alpha", mu=0.5, sigma=0.3)      # spin-down index
    beta      = pm.Normal("beta",  mu=0.0, sigma=1.0)      # log-scale intercept
    sigma_int = pm.HalfNormal("sigma_int", sigma=0.3)      # intrinsic scatter

    # ── Nuisance: true ages (latent) ────────────────────
    log_age_true = pm.Normal("log_age_true",
                             mu=log_age_obs, sigma=sigma_age_obs,
                             shape=N)

    # ── Physical model ──────────────────────────────────
    log_P_pred = alpha * log_age_true + beta

    # ── Likelihood ──────────────────────────────────────
    log_P_obs_like = pm.Normal("log_P_obs",
                               mu=log_P_pred,
                               sigma=np.sqrt(sigma_P_obs**2 + sigma_int**2),
                               observed=log_P_rot_obs)

    # ── Sample ──────────────────────────────────────────
    trace = pm.sample(2000, tune=1000, target_accept=0.95,
                      nuts_sampler="numpyro")  # fast JAX backend

# Extract population results
print(pm.summary(trace, var_names=["alpha", "beta", "sigma_int"]))
```

---

## 8.6 Application: The Exoplanet Mass-Radius Relation

**Population model:** Power law with intrinsic scatter

    R_true = c * M_true^alpha * exp(eps),  eps ~ N(0, sigma_int^2)

**Measurement model:** Both mass and radius measured with Gaussian noise

    M_obs,i ~ N(M_true,i, sigma_M,i^2)
    R_obs,i ~ N(R_true,i, sigma_R,i^2)

This is a **regression with errors in both variables** — properly handled by the hierarchical latent-variable model. Classical linear regression assumes X is error-free and gives biased results.

**Physical insight gained:**
- alpha ≈ 0.55 for sub-Neptune planets (radius increases slowly with mass)
- alpha ≈ 0.01 for gas giants (roughly constant radius)
- sigma_int encodes compositional diversity at fixed mass

---

## 8.7 Cosmological Application: H0 from Gravitational Waves

Standard sirens: binary neutron star mergers are "standard sirens" — their luminosity distance D_L can be inferred from the GW signal without any calibration. Combined with host galaxy redshift z, they constrain H_0:

    D_L = (c/H_0) * (1+z) * integral_0^z dz' / E(z')

**Hierarchical model:**
- Observe N GW events, each with posterior P(D_L^i, inclination^i | GW data)
- Each host galaxy has measured redshift z_i with uncertainty
- H_0, Omega_m, w are shared population parameters

Marginalise over inclination and distance posteriors per event → combine into a joint H_0 posterior. This was done with GW170817, yielding H_0 = 70 +12/-8 km/s/Mpc.

---

## 8.8 Non-Parametric Population Models

When the functional form of the population is unknown:

**Gaussian Mixture Models:**
    P(theta | phi) = sum_k pi_k * N(theta | mu_k, Sigma_k)

**Dirichlet Process (infinite mixture):** Non-parametric prior over mixtures — the number of components is learned from data. Implemented in PyMC with `pm.Dirichlet`.

**Normalising Flows:** Neural network-based flexible density estimators for the population — state of the art for gravitational wave population inference.

---

## 8.9 Computational Considerations

| Method | When to use |
|--------|-------------|
| Gibbs sampling | Conjugate conditionals available |
| NUTS/HMC (PyMC, Stan) | Differentiable model, d < 1000 |
| emcee + importance sampling | Non-differentiable, moderate N |
| Variational inference | Large N, approximate posteriors acceptable |
| Sequential Monte Carlo | Very high-dimensional population |

**Key hyperparameter:** `target_accept=0.95` in PyMC for hierarchical models — the geometry is often funnel-shaped, requiring small steps (high acceptance).

**Non-centred parameterisation:** For deep hierarchies, centred parameterisation causes sampling pathologies. Use:

    theta_i = mu + tau * z_i,  z_i ~ N(0, 1)   [non-centred]

instead of theta_i ~ N(mu, tau^2). This removes the correlation between individual parameters and the hyperparameters.

---

## 8.10 Summary

Hierarchical models:
1. **Borrow statistical strength** across objects through the shared population model
2. **Shrink** noisy individual estimates toward the population
3. **Properly propagate** measurement uncertainties at all levels
4. **Infer population distributions** from heterogeneous, noisy data
5. Are the **standard approach** in gravitational wave astronomy, exoplanet demographics, and survey cosmology

---

## 📝 Final Synthesis

Design and implement a hierarchical model for the following:

**Data:** 30 Type Ia supernovae with measured peak magnitude m_i, redshift z_i, and colour c_i. Each has measurement uncertainties sigma_m,i, sigma_z,i, sigma_c,i.

**Goal:** Infer H_0, the absolute magnitude M, and the colour correction coefficient beta.

**Model:**
- mu_i = 5 log10(D_L(z_i, H_0)) + 25   (distance modulus)
- m_i_pred = M + mu_i + beta * c_i      (standardised candle)
- m_i_obs ~ N(m_i_pred, sigma_m,i^2 + sigma_int^2)

Use PyMC with NUTS. Report the posterior on H_0 with 68% and 95% credible intervals.

---

## 🔗 Navigation
[← Chapter 7: Gaussian Processes](../ch07_gaussian_processes/README.md)  
[→ Capstone Project: Exoplanet Transit Analysis](../../project/exoplanet_transit/README.md)
