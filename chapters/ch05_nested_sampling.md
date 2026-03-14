# Chapter 5: Nested Sampling

> *"The evidence is everything."* — Bayesian Model Comparison

---

## 5.1 Motivation: Computing the Evidence

MCMC samples the posterior but does not directly compute the **Bayesian evidence**:
$$Z = P(D \mid \mathcal{M}) = \int \mathcal{L}(\theta) \pi(\theta)\, d\theta$$

Nested sampling computes Z directly, making it indispensable for **model comparison**.

---

## 5.2 The Algorithm

The key insight: transform the multi-dimensional integral into a 1D integral.

**Define the prior volume:**
$$X(\lambda) = \int_{\mathcal{L}(\theta) > \lambda} \pi(\theta)\, d\theta$$

This is the fraction of prior volume where the likelihood exceeds λ. Note:
- $X(0) = 1$ (all prior volume has likelihood > 0)
- $X(\mathcal{L}_{\max}) \to 0$

The evidence becomes:
$$\boxed{Z = \int_0^1 \mathcal{L}(X)\, dX}$$

**Algorithm (Skilling 2004):**

1. Draw N **live points** from the prior: $\{\theta_i\}_{i=1}^N$
2. Find the point with lowest likelihood: $\mathcal{L}_{\min}$
3. Replace it with a new point drawn from prior, constrained to $\mathcal{L} > \mathcal{L}_{\min}$
4. The discarded point contributes to the evidence sum
5. Repeat until $\Delta \ln Z < \epsilon$

**Evidence estimate:**
$$\ln Z \approx \ln \sum_i w_i \mathcal{L}_i, \quad w_i = X_{i-1} - X_i \approx e^{-i/N} - e^{-(i+1)/N}$$

---

## 5.3 MultiNest and dynesty

**MultiNest** (Feroz & Hobson 2009): Uses ellipsoidal decomposition to efficiently sample constrained regions. Standard in X-ray (XSPEC) and gravitational wave astronomy.

**dynesty** (Speagle 2020): Dynamic nested sampling — adaptively allocates live points. Pure Python, integrates with astrophysics ecosystem.

```python
import dynesty

def log_likelihood(theta):
    return -0.5 * chi_squared(theta, data)

def prior_transform(u):
    # Map unit hypercube u ∈ [0,1]^d to physical parameters
    theta = np.zeros_like(u)
    theta[0] = u[0] * 20 - 10      # Uniform [-10, 10]
    theta[1] = stats.norm.ppf(u[1], loc=0, scale=5)  # Gaussian
    return theta

sampler = dynesty.NestedSampler(log_likelihood, prior_transform, ndim=2)
sampler.run_nested()
results = sampler.results

print(f"ln Z = {results.logz[-1]:.2f} ± {results.logzerr[-1]:.2f}")
```

---

## 5.4 Posterior from Nested Sampling

Nested sampling also yields posterior samples with weights:
$$w_i \propto \mathcal{L}_i \cdot (X_{i-1} - X_i)$$

Resample with replacement using weights to get unweighted posterior samples.

---

## 📝 Exercises

1. Run `dynesty` on a 2D Gaussian likelihood with a uniform prior. Compare the evidence to the analytic value.
2. Compare posterior from `dynesty` vs `emcee` for a simple spectral fitting problem. Are they consistent?

---

# Chapter 6: Bayesian Model Comparison

---

## 6.1 The Bayes Factor

Given two models $\mathcal{M}_1$ and $\mathcal{M}_2$, the **Bayes factor** is:

$$\boxed{B_{12} = \frac{P(D \mid \mathcal{M}_1)}{P(D \mid \mathcal{M}_2)} = \frac{Z_1}{Z_2}}$$

Combined with model priors:
$$\frac{P(\mathcal{M}_1 \mid D)}{P(\mathcal{M}_2 \mid D)} = B_{12} \cdot \frac{P(\mathcal{M}_1)}{P(\mathcal{M}_2)}$$

**Jeffreys scale for interpreting B₁₂:**

| $\ln B_{12}$ | Interpretation |
|-------------|----------------|
| < 1 | Not worth mentioning |
| 1 – 2.5 | Substantial evidence for $\mathcal{M}_1$ |
| 2.5 – 5 | Strong evidence |
| > 5 | Decisive evidence |

---

## 6.2 Occam's Razor is Automatic

The Bayesian evidence automatically penalises complex models:
$$Z = \int \mathcal{L}(\theta) \pi(\theta)\, d\theta \approx \mathcal{L}_{\max} \cdot \underbrace{\frac{\Delta\theta_{\text{posterior}}}{\Delta\theta_{\text{prior}}}}_{\text{Occam factor}}$$

A model with more parameters has a smaller Occam factor unless the likelihood strongly supports those parameters. **No hand-tuning of penalty terms required.**

---

## 6.3 Information Criteria (Approximations)

When nested sampling is too expensive:

**AIC (Akaike Information Criterion):**
$$\text{AIC} = -2 \ln \mathcal{L}_{\max} + 2k$$

**BIC (Bayesian Information Criterion):**
$$\text{BIC} = -2 \ln \mathcal{L}_{\max} + k \ln N$$

**DIC (Deviance Information Criterion):**
$$\text{DIC} = \overline{D(\theta)} + p_D, \quad p_D = \overline{D(\theta)} - D(\bar{\theta})$$

where $D(\theta) = -2\ln\mathcal{L}(\theta)$ is the deviance. DIC uses the full posterior.

**WAIC (Widely Applicable IC):** Most reliable approximation for hierarchical models.

---

## 6.4 Application: Line Detection in a Spectrum

**$\mathcal{M}_0$:** Continuum only → $Z_0$  
**$\mathcal{M}_1$:** Continuum + emission line → $Z_1$

$\ln B_{10} = \ln Z_1 - \ln Z_0 > 5$ → decisive evidence for the line.

---

# Chapter 7: Gaussian Processes

---

## 7.1 What is a Gaussian Process?

A **Gaussian Process (GP)** is a probability distribution over **functions**:

$$f(\mathbf{x}) \sim \mathcal{GP}(m(\mathbf{x}),\, k(\mathbf{x}, \mathbf{x}'))$$

Any finite collection of function values is jointly Gaussian:
$$\mathbf{f} = [f(x_1), \ldots, f(x_n)]^\top \sim \mathcal{N}(\mathbf{m}, \mathbf{K})$$

where $K_{ij} = k(x_i, x_j)$ is the **covariance (kernel) matrix**.

---

## 7.2 Kernels

The kernel encodes assumptions about the function's smoothness:

**Squared Exponential (RBF):**
$$k_{\text{SE}}(x, x') = \sigma_f^2 \exp\!\left(-\frac{(x-x')^2}{2\ell^2}\right)$$

→ Infinitely smooth, good for slowly varying astrophysical signals.

**Matérn 3/2:**
$$k_{3/2}(r) = \sigma_f^2\!\left(1 + \frac{\sqrt{3}r}{\ell}\right)\exp\!\left(-\frac{\sqrt{3}r}{\ell}\right), \quad r = |x-x'|$$

→ Less smooth, better for stellar variability, transients.

**Quasi-Periodic:**
$$k_{\text{QP}}(x,x') = \sigma_f^2 \exp\!\left(-\frac{(x-x')^2}{2\ell^2} - \frac{2\sin^2\!\left(\pi(x-x')/P\right)}{\alpha^2}\right)$$

→ Models stellar rotation with period P.

---

## 7.3 GP Regression

Given noisy observations $\mathbf{y} = \mathbf{f} + \boldsymbol{\epsilon}$ with $\epsilon_i \sim \mathcal{N}(0, \sigma_n^2)$:

**Posterior predictive at new point x***:
$$\mu_* = \mathbf{k}_*^\top (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{y}$$
$$\sigma_*^2 = k_{**} - \mathbf{k}_*^\top (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{k}_*$$

**Log marginal likelihood for hyperparameter optimisation:**
$$\ln P(\mathbf{y} \mid \mathbf{X}, \phi) = -\frac{1}{2}\mathbf{y}^\top C^{-1}\mathbf{y} - \frac{1}{2}\ln|C| - \frac{n}{2}\ln 2\pi$$

where $C = \mathbf{K} + \sigma_n^2 \mathbf{I}$ and $\phi = (\sigma_f, \ell, \sigma_n)$.

**Cost:** $O(n^3)$ for matrix inversion — a major limitation for large datasets.

---

## 7.4 GP in Astrophysics

**Applications:**
- Stellar light curve modelling (George package, celerite for O(n) speed)
- Interpolating spectra over masked regions
- Modelling correlated noise in radial velocity data
- Redshift-distance relation interpolation

```python
import george
from george import kernels

kernel = kernels.ExpSquaredKernel(metric=0.5)
gp = george.GP(kernel)
gp.compute(t, yerr)

mu, var = gp.predict(y, t_pred, return_var=True)
```

---

# Chapter 8: Hierarchical Bayesian Models

---

## 8.1 The Population Inference Problem

We observe N objects, each with noisy data $D_i$. We want to infer:
1. Individual parameters $\theta_i$ for each object
2. Population-level parameters $\phi$ describing the distribution of $\theta_i$

**Naive approach:** Fit each object separately → misses correlations, ignores population structure.

**Hierarchical approach:** Model them jointly.

---

## 8.2 The Hierarchical Model

$$\phi \sim P(\phi) \quad \text{(hyperprior)}$$
$$\theta_i \sim P(\theta_i \mid \phi) \quad \text{(population model)}$$
$$D_i \sim P(D_i \mid \theta_i) \quad \text{(likelihood)}$$

Joint posterior:
$$\boxed{P(\phi, \{\theta_i\} \mid \{D_i\}) \propto P(\phi) \prod_{i=1}^N P(D_i \mid \theta_i) P(\theta_i \mid \phi)}$$

---

## 8.3 Marginalising Over Individual Parameters

If individual parameters $\theta_i$ are not of direct interest:
$$P(\phi \mid \{D_i\}) \propto P(\phi) \prod_{i=1}^N \int P(D_i \mid \theta_i) P(\theta_i \mid \phi)\, d\theta_i$$

Each integral is called the **marginal likelihood** for object i.

---

## 8.4 Shrinkage: Borrowing Strength

Hierarchical models exhibit **partial pooling**: estimates for individual objects are "shrunk" toward the population mean. This is statistically optimal — it reduces mean squared error compared to fitting each object independently.

**Stein's paradox:** For d ≥ 3, shrinkage estimators dominate MLE estimators in terms of MSE.

---

## 8.5 Astrophysical Application: The Mass-Radius Relation of Exoplanets

**Setup:** N exoplanets, each with noisy mass $M_i^{\text{obs}}$ and radius $R_i^{\text{obs}}$ measurements.

**Model:** Power-law relation $R = c M^\alpha + \epsilon$

$$\phi = (\alpha, c, \sigma_{\text{int}}) \quad \text{(population)}$$
$$M_i^{\text{true}}, R_i^{\text{true}} \sim P(\cdot \mid \alpha, c, \sigma_{\text{int}}) \quad \text{(individual truth)}$$
$$M_i^{\text{obs}} \sim \mathcal{N}(M_i^{\text{true}}, \sigma_{M,i}^2), \quad R_i^{\text{obs}} \sim \mathcal{N}(R_i^{\text{true}}, \sigma_{R,i}^2) \quad \text{(measurement)}$$

This model:
- Accounts for measurement noise in *both* variables
- Infers the intrinsic scatter $\sigma_{\text{int}}$
- Shares statistical strength across all planets

---

## 8.6 Implementation: PyMC Example

```python
import pymc as pm

with pm.Model() as hierarchical_model:
    # Hyperpriors
    mu_alpha = pm.Normal("mu_alpha", 0, 10)
    sigma_alpha = pm.HalfNormal("sigma_alpha", 1)
    
    # Individual parameters
    alpha = pm.Normal("alpha", mu_alpha, sigma_alpha, shape=N)
    
    # Likelihood
    y_hat = pm.math.dot(X, alpha)
    sigma = pm.HalfNormal("sigma", 1)
    y_obs = pm.Normal("y_obs", y_hat, sigma, observed=y)
    
    # Sample
    trace = pm.sample(2000, tune=1000, target_accept=0.95)
```

---

## 📝 Final Synthesis Exercise

Design a hierarchical model for a sample of 50 Type Ia supernovae. Each has a measured peak magnitude $m_i$ and redshift $z_i$ with uncertainties. Infer $H_0$ and $M_{\text{abs}}$ simultaneously while marginalising over peculiar velocities modelled as a Gaussian with unknown dispersion.

---

## 🔗 Navigation

[← Chapter 7: Gaussian Processes](../ch07_gaussian_processes/README.md)  
[→ Capstone Project: Exoplanet Transit Analysis](../../project/exoplanet_transit/README.md)
