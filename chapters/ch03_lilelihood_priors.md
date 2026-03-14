# Chapter 3: Likelihood Functions and Prior Selection

> *"The likelihood is the bridge between data and parameters."*

---

## 3.1 The Likelihood Function

Given a statistical model with parameters θ and observed data D = {d₁, d₂, ..., dₙ}, the **likelihood function** is:

$$\boxed{\mathcal{L}(\theta) \equiv P(D \mid \theta) = \prod_{i=1}^n P(d_i \mid \theta)}$$

(The last equality assumes the data points are **conditionally independent** given θ.)

**Critical distinction:**
- $P(D \mid \theta)$: viewed as a function of D for fixed θ → probability distribution (integrates to 1)
- $\mathcal{L}(\theta) = P(D \mid \theta)$: viewed as a function of θ for fixed D → likelihood function (does NOT integrate to 1)

---

## 3.2 The Log-Likelihood

Working in log space is numerically essential (products of small numbers underflow):

$$\ln \mathcal{L}(\theta) = \sum_{i=1}^n \ln P(d_i \mid \theta)$$

**MLE is found by:**
$$\hat{\theta}_{\text{MLE}} = \arg\max_\theta \ln \mathcal{L}(\theta)$$

Setting $\frac{\partial \ln \mathcal{L}}{\partial \theta} = 0$ and solving.

---

## 3.3 Constructing the Likelihood from Physics

The likelihood encodes your **noise model** — how measurements differ from true values.

### 3.3.1 Gaussian Noise (Spectroscopy, Photometry at High Counts)

Observed flux $F_i^{\text{obs}}$ at wavelength $\lambda_i$, with known uncertainty $\sigma_i$:

$$P(F_i^{\text{obs}} \mid \theta) = \frac{1}{\sigma_i \sqrt{2\pi}} \exp\!\left(-\frac{(F_i^{\text{obs}} - F_i^{\text{model}}(\theta))^2}{2\sigma_i^2}\right)$$

Log-likelihood:
$$\ln \mathcal{L}(\theta) = -\frac{1}{2} \sum_{i=1}^n \left[\frac{(F_i^{\text{obs}} - F_i^{\text{model}}(\theta))^2}{\sigma_i^2} + \ln(2\pi\sigma_i^2)\right]$$

$$= -\frac{1}{2}\chi^2(\theta) + \text{const}$$

where $\chi^2(\theta) = \sum_i (F_i^{\text{obs}} - F_i^{\text{model}}(\theta))^2 / \sigma_i^2$.

**MLE of a Gaussian model = minimising χ²** — least squares is a special case of Bayesian inference with Gaussian noise and flat priors!

### 3.3.2 Poisson Likelihood (Low Count X-ray, Gamma-ray)

Observed counts $n_i$ in bin i, model predicts $\mu_i(\theta)$ expected counts:

$$P(n_i \mid \theta) = \frac{\mu_i(\theta)^{n_i} e^{-\mu_i(\theta)}}{n_i!}$$

Log-likelihood:
$$\boxed{\ln \mathcal{L}(\theta) = \sum_{i=1}^n \left[n_i \ln \mu_i(\theta) - \mu_i(\theta) - \ln(n_i!)\right]}$$

The Cash statistic (C-stat) used in X-ray astronomy is: $C = -2 \ln \mathcal{L}$.

### 3.3.3 Mixed / Heteroscedastic Noise

When uncertainties vary and the noise model itself has uncertain parameters (e.g., unknown systematic floor $s$):

$$\sigma_{\text{eff},i}^2 = \sigma_i^2 + s^2$$

$$\ln \mathcal{L}(\theta, s) = -\frac{1}{2}\sum_i \left[\frac{(d_i - m_i(\theta))^2}{\sigma_i^2 + s^2} + \ln(\sigma_i^2 + s^2)\right]$$

Now s is a **nuisance parameter** that we marginalise over.

---

## 3.4 Fisher Information and the Cramér-Rao Bound

**Fisher Information Matrix:**
$$\mathcal{F}_{ij}(\theta) = -E\left[\frac{\partial^2 \ln \mathcal{L}}{\partial \theta_i \partial \theta_j}\right]$$

The **Cramér-Rao lower bound:**
$$\text{Cov}(\hat{\theta}) \geq \mathcal{F}^{-1}(\theta)$$

No unbiased estimator can have variance smaller than the inverse Fisher information. This sets the fundamental precision limit for parameter estimation.

**In astrophysics:** Used to forecast parameter constraints for future experiments (LSST, SKA, LISA) before they are built.

---

## 3.5 Prior Distributions: Types and Choices

The prior $P(\theta)$ encodes knowledge *before* seeing the data.

### 3.5.1 Uninformative Priors

**Flat (uniform) prior:**
$$P(\theta) \propto 1 \quad \text{for } \theta \in [a, b]$$

Warning: Not truly "uninformative" — flat in θ is NOT flat in g(θ). Choosing flat prior on mass ≠ flat prior on log(mass).

**Jeffreys prior** (invariant under reparametrisation):
$$\boxed{P_J(\theta) \propto \sqrt{\det \mathcal{F}(\theta)}}$$

For a Gaussian likelihood: $P_J(\sigma) \propto 1/\sigma$ (log-flat on scale parameters).

For a Poisson rate λ: $P_J(\lambda) \propto \lambda^{-1/2}$

### 3.5.2 Log-Uniform Prior (Scale Parameters)

When a parameter spans many orders of magnitude (e.g., period of a pulsar, luminosity of a galaxy):
$$P(\theta) \propto \frac{1}{\theta} \quad \Leftrightarrow \quad P(\ln\theta) = \text{const}$$

This is the Jeffreys prior for scale parameters.

### 3.5.3 Informative Priors from Physical Constraints

**Example: Black hole spin parameter a ∈ [0, 1)**

Physical theory (no naked singularities) gives $0 \leq a < 1$. We encode:
$$P(a) = \begin{cases} 2a & 0 \leq a \leq 1 \\ 0 & \text{otherwise} \end{cases}$$

(Linearly increasing reflects that rapidly spinning BHs are more common in simulations.)

**Example: Stellar mass from Initial Mass Function (IMF)**

Salpeter IMF: $P(M) \propto M^{-2.35}$ for $M > 0.1 M_\odot$.

Kroupa IMF (piecewise):
$$P(M) \propto \begin{cases} M^{-1.3} & 0.08 \leq M/M_\odot < 0.5 \\ M^{-2.3} & M/M_\odot \geq 0.5 \end{cases}$$

### 3.5.4 Empirical Priors (Hierarchical)

Use observational constraints from the literature as Gaussian priors:
$$P(\theta) = \mathcal{N}(\mu_{\text{lit}}, \sigma_{\text{lit}}^2)$$

E.g., Hubble constant prior from CMB: $H_0 \sim \mathcal{N}(67.4, 0.5^2)$ km/s/Mpc.

---

## 3.6 Prior Sensitivity Analysis

A good Bayesian analysis should test whether conclusions depend strongly on the prior choice.

**Protocol:**
1. Run inference with your primary prior
2. Repeat with 2-3 alternative priors (different widths, functional forms)
3. Compare posteriors: if they agree, conclusions are prior-insensitive (data-dominated)
4. If they disagree, more data or stronger physical arguments are needed

---

## 3.7 The Posterior in Full

Putting it together: for a spectral fitting problem with parameters $\theta = (T_{\text{eff}}, \log g, [\text{Fe/H}])$ (temperature, surface gravity, metallicity):

$$\underbrace{P(\theta \mid D)}_{\text{posterior}} \propto \underbrace{\prod_{i=1}^N \exp\!\left(-\frac{(F_i^{\text{obs}} - F_i^{\text{model}}(\theta))^2}{2\sigma_i^2}\right)}_{\text{Gaussian likelihood}} \times \underbrace{P(T_{\text{eff}}) \cdot P(\log g) \cdot P([\text{Fe/H}])}_{\text{factored prior (independence assumed)}}$$

---

## 3.8 Nuisance Parameters and Marginalisation

Often we care about subset $\theta = (\theta_{\text{interest}}, \theta_{\text{nuisance}})$.

**Marginalise** over nuisance parameters:
$$\boxed{P(\theta_{\text{interest}} \mid D) = \int P(\theta_{\text{interest}}, \theta_{\text{nuisance}} \mid D)\, d\theta_{\text{nuisance}}}$$

This automatically propagates all uncertainty from nuisance parameters into the inference on parameters of interest. It is one of the most powerful aspects of Bayesian analysis.

**Example:** Measuring a pulsar's dispersion measure (DM) while the distance d is unknown but uncertain. Integrate over d to get marginal posterior on DM.

---

## 📝 Exercises

1. **Likelihood construction:** Design a likelihood for an X-ray spectrum with 50 energy bins where some bins have < 5 counts. Should you use Gaussian or Poisson? Write out the full expression.

2. **Jeffreys prior:** Derive the Jeffreys prior for a Binomial likelihood $P(k|n,p) = \binom{n}{k}p^k(1-p)^{n-k}$. (Hint: compute the Fisher information for p.)

3. **Prior sensitivity:** Generate synthetic data from N(μ=5, σ=1) with n=10 points. Infer μ analytically using: (a) flat prior, (b) N(0,1) prior, (c) N(10,1) prior. When does the data dominate over the prior?

4. **Marginalisation:** A spectral line has amplitude A and background level B (nuisance). Write the joint posterior P(A, B | D) and show how to compute the marginal P(A | D).

---

## 🔗 Navigation

[← Chapter 2: Bayes' Theorem](../ch02_bayes_theorem/README.md)  
[→ Chapter 4: MCMC Methods](../ch04_mcmc/README.md)