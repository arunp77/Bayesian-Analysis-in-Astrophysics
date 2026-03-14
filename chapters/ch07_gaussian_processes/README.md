# Chapter 7: Gaussian Processes

> *"A GP is a distribution over functions, not just over parameter values."*

---

## 7.1 What is a Gaussian Process?

A Gaussian Process (GP) is a probability distribution over **functions**:

    f(x) ~ GP(m(x), k(x, x'))

Any finite collection of function values is jointly Gaussian:

    f = [f(x_1), ..., f(x_n)]^T  ~  N(m, K)

where K_ij = k(x_i, x_j) is the **kernel matrix**.

**Why GPs in astrophysics?**  
Stellar light curves have correlated noise from activity (spots, oscillations). Radial velocity data have correlated systematics. A GP models these correlations explicitly, preventing them from biasing planet/signal parameters.

---

## 7.2 Kernel Functions

The kernel encodes assumptions about function smoothness:

### Squared Exponential (RBF)

    k_SE(x, x') = sigma_f^2 * exp(-(x-x')^2 / (2 ell^2))

→ Infinitely differentiable. Good for slowly-varying signals.  
Hyperparameters: amplitude sigma_f, length scale ell.

### Matérn 3/2

    k_3/2(r) = sigma_f^2 * (1 + sqrt(3)*r/ell) * exp(-sqrt(3)*r/ell)

where r = |x - x'|. Less smooth than RBF, better for stellar variability.

### Quasi-Periodic (stellar rotation)

    k_QP(x, x') = sigma_f^2 * exp(-(x-x')^2/(2*ell^2) - 2*sin^2(pi*(x-x')/P) / alpha^2)

Models periodic variability with period P and decay timescale ell. Standard for modelling stellar rotation in radial velocity analysis.

### Combining Kernels

Kernels can be added and multiplied:
- **Sum:** k = k_1 + k_2 → models independent processes
- **Product:** k = k_1 × k_2 → non-stationary modulation

---

## 7.3 GP Regression

Given noisy observations y = f(x) + epsilon, with epsilon_i ~ N(0, sigma_n^2):

**Posterior predictive at new point x***:

    mu_* = k_*^T (K + sigma_n^2 I)^{-1} y
    sigma_*^2 = k(x*, x*) - k_*^T (K + sigma_n^2 I)^{-1} k_*

where k_* = [k(x*, x_1), ..., k(x*, x_n)]^T.

**Log marginal likelihood** (for optimising hyperparameters phi = (sigma_f, ell, sigma_n)):

    ln P(y | X, phi) = -1/2 * y^T C^{-1} y - 1/2 * ln|C| - n/2 * ln(2 pi)

where C = K + sigma_n^2 I.

This can be maximised (empirical Bayes) or the hyperparameters can be fully marginalised via MCMC.

**Computational cost:** O(n^3) for matrix inversion — scales badly. For n > 10^4, use approximations.

---

## 7.4 celerite: Fast O(n) GPs for Time Series

For 1D data with separable kernels, the Cholesky factorisation has O(n) cost (Foreman-Mackey et al. 2017). The `celerite2` package implements this:

```python
import celerite2
from celerite2 import terms

# Quasi-periodic kernel (Rotation term)
term = terms.RotationTerm(
    sigma=0.1,     # amplitude
    period=10.0,   # rotation period (days)
    Q0=0.5,        # quality factor
    dQ=0.5,        # differential rotation
    f=0.5          # fractional secondary amplitude
)

gp = celerite2.GaussianProcess(term, mean=0.0)
gp.compute(t, yerr=yerr)

log_prob = gp.log_likelihood(y - model(theta))
mu, var = gp.predict(y - model(theta), t=t_pred, return_var=True)
```

---

## 7.5 Joint Transit + GP Fit

For exoplanet transit data with stellar noise, the full likelihood is:

    y_i = F_transit(t_i; theta_transit) + GP_noise(t_i; phi)

**Log likelihood:**

    ln L(theta, phi) = -1/2 * r^T C^{-1} r - 1/2 * ln|C| - n/2 ln(2pi)

where r_i = y_i - F_transit(t_i; theta_transit) are the residuals.

The transit parameters theta and GP hyperparameters phi are sampled jointly with emcee or dynesty. This correctly marginalises over stellar noise uncertainty.

---

## 7.6 Radial Velocity Analysis

The radial velocity (RV) signal from an orbiting planet:

    RV(t) = K * [cos(v(t) + omega) + e*cos(omega)] + gamma

where K is the semi-amplitude, v(t) is the true anomaly, e is eccentricity, omega is argument of periastron, gamma is the systemic velocity.

Stellar activity (spots, granulation) creates correlated RV noise. A GP with a quasi-periodic kernel models this, allowing extraction of planetary signals even from active stars.

Standard workflow:
1. Fit GP + Keplerian model jointly
2. Marginalise over GP hyperparameters
3. Report posterior on (K, P, e, omega)
4. Derive minimum planet mass: m sin(i) from K, P, stellar mass

---

## 7.7 Hyperparameter Optimisation Strategies

**Maximum Likelihood II (MLII / Empirical Bayes):**
```python
from scipy.optimize import minimize
def neg_log_ml(log_params):
    sigma_f, ell, sigma_n = np.exp(log_params)
    gp.set_params(sigma_f=sigma_f, ell=ell, sigma_n=sigma_n)
    return -gp.log_marginal_likelihood(y)

result = minimize(neg_log_ml, x0=[0, 0, -2], method='L-BFGS-B')
```

**Full Bayes (preferred):** Place priors on hyperparameters and sample with MCMC. Propagates hyperparameter uncertainty into predictions.

---

## 📝 Exercises

1. **GP regression from scratch:** Implement GP regression with an RBF kernel in pure NumPy. Fit to 20 noisy sine wave points, vary ell, and observe how the posterior predictive changes.

2. **celerite warmup:** Download a TESS light curve of a magnetically active star. Fit a GP with a RotationTerm. What is the inferred rotation period?

3. **Joint fit:** Simulate a transit light curve with added correlated noise from a Matérn kernel. Fit with: (a) transit only, (b) transit + GP. Compare parameter posteriors and residuals.

4. **Model comparison:** Compute the Bayes factor between a white-noise-only model and a GP+white-noise model for a set of radial velocity observations.

---

## 🔗 Navigation
[← Chapter 6: Model Comparison](../ch06_model_comparison/README.md)  
[→ Chapter 8: Hierarchical Models](../ch08_hierarchical_models/README.md)
