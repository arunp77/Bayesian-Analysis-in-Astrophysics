# 🧠 1. Priors (belief before seeing data)

### Uniform prior

$$
p(x) = \frac{1}{\text{high} - \text{low}}, \quad x \in [\text{low}, \text{high}]
$$

$$
\log p(x) = -\log(\text{high} - \text{low})
$$

👉 **Use case**

- Non-informative prior when all values in a range are equally likely
- Example: slope (m \in [-10, 10])

---

### Log-uniform (Jeffreys prior)

$$
p(x) \propto \frac{1}{x}
$$

$$
\log p(x) = -\log x - \log\left(\log\frac{\text{high}}{\text{low}}\right)
$$

👉 **Use case**

- Scale parameters (variance, noise)
- Invariant under scaling
- Common in astrophysics & Bayesian inference

---

### Gaussian prior

$$
p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

$$
\log p(x) = -\frac{(x-\mu)^2}{2\sigma^2} - \frac{1}{2}\log(2\pi\sigma^2)
$$

👉 **Use case**

- When you have prior knowledge centered around a value
- Example: known calibration parameter

---

### Half-Gaussian prior (positive-only)

$$
p(x) \propto \exp\left(-\frac{x^2}{2\sigma^2}\right), \quad x > 0
$$

👉 **Use case**

- Noise scale, variance, jitter
- Ensures parameter stays positive

---

# 📊 2. Likelihood functions (fit to data)

### Gaussian likelihood

$$
\log L = -\frac{1}{2} \sum_i \left[ \frac{(d_i - m_i)^2}{\sigma_i^2} + \log(2\pi \sigma_i^2) \right]
$$

👉 **Use case**

- Continuous data with Gaussian noise
- Most common in regression, physics measurements

---

### Poisson likelihood

$$
\log L = \sum_i \left[ k_i \log \lambda_i - \lambda_i - \log(k_i!) \right]
$$

👉 **Use case**

- Count data (events, photons, arrivals)
- Common in:
  - particle physics
  - astronomy
  - event logs

---

### Heteroscedastic Gaussian (with jitter)

$$
\sigma_{\text{eff}}^2 = \sigma_{\text{obs}}^2 + \sigma_{\text{jitter}}^2
$$

👉 **Use case**

- When measurement errors are underestimated
- Adds unknown systematic noise
- Very practical for real-world data

---

# 🔁 3. Bayesian inference (core idea)

All your functions combine into:

$$
\log p(\theta | D) = \log p(\theta) + \log p(D | \theta)
$$

👉 **Use case**

- This is the **posterior** used in MCMC
- In the bayes_utils.py, `log_posterior()` implements exactly this

---

# 🔄 4. MCMC diagnostics

### Gelman–Rubin statistic (R-hat)

$$
\hat{R} = \sqrt{\frac{\hat{V}}{W}}
$$

Where:

- $W$: within-chain variance
- $B$: between-chain variance
- $\hat{V}$: combined estimate

👉 **Use case**

- Check convergence of multiple chains
- Rule of thumb:
  - $ \hat{R} < 1.01 $ → good

---

### Effective Sample Size (ESS)

$$
\text{ESS} = \frac{N}{\tau}
$$

$$
\tau = 1 + 2 \sum_{k=1}^{\infty} \rho_k
$$

👉 **Use case**

- Measures how many _independent_ samples you really have
- Important because MCMC samples are correlated

---

# 📈 5. Posterior summary

For parameter ( \theta ):

- Median:

  $$
  \tilde{\theta} = \text{median}(\theta)
  $$

- Credible interval:
  $$
  [\theta_{\alpha/2}, \theta_{1-\alpha/2}]
  $$

👉 **Use case**

- Bayesian alternative to confidence intervals
- Direct probabilistic interpretation

---

# ⚖️ 6. Model comparison

### Deviance Information Criterion (DIC)

$$
\text{DIC} = \bar{D} + p_D
$$

$$
p_D = \bar{D} - D(\bar{\theta})
$$

👉 **Use case**

- Bayesian model comparison
- Penalizes model complexity

---

### AIC (Akaike Information Criterion)

$$
\text{AIC} = -2 \log L_{\max} + 2k
$$

---

### BIC (Bayesian Information Criterion)

$$
\text{BIC} = -2 \log L_{\max} + k \log n
$$

👉 **Use case**

- Compare models:
  - Lower = better

- AIC → predictive accuracy
- BIC → stronger penalty (favours simpler models)

---

# 🚀 7. Your example (linear regression model)

Model:

$$
y = m x + b + \epsilon
$$

Noise:

$$
\epsilon \sim \mathcal{N}(0, \sigma^2 + s^2)
$$

👉 **What you're doing**

- Estimating:
  - slope (m)
  - intercept (b)
  - extra noise (s)

- Using:
  - uniform priors
  - Gaussian likelihood
  - MCMC (via `emcee`)

---

# 🧩 Big picture

Your file implements the full Bayesian pipeline:

1. **Define prior** → belief
2. **Define likelihood** → data fit
3. **Compute posterior** → inference
4. **Sample with MCMC** → explore parameter space
5. **Diagnose convergence** → reliability
6. **Summarize & compare models** → decision making
