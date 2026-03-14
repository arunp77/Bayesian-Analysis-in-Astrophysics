# Chapter 6: Bayesian Model Comparison

> *"All models are wrong, but some are useful — and Bayes tells you which."*

---

## 6.1 The Model Comparison Problem

After fitting a model, we face a deeper question: **is this the right model?**

- Does a spectral line actually exist, or is it noise?
- Is this light curve variation intrinsic, or instrumental?
- Does the data require a broken power law, or is a simple power law sufficient?

Classical approaches (F-test, likelihood ratio test) make assumptions that break down for nested and non-nested models alike. The Bayesian solution is elegant and general.

---

## 6.2 The Bayes Factor

Given two models M_1 and M_2:

$$\frac{P(\mathcal{M}_1 \mid D)}{P(\mathcal{M}_2 \mid D)} = \underbrace{\frac{P(D \mid \mathcal{M}_1)}{P(D \mid \mathcal{M}_2)}}_{B_{12}} \times \frac{P(\mathcal{M}_1)}{P(\mathcal{M}_2)}$$

$$\boxed{B_{12} = \frac{Z_1}{Z_2} = \frac{\int \mathcal{L}_1(\theta_1)\, \pi_1(\theta_1)\, d\theta_1}{\int \mathcal{L}_2(\theta_2)\, \pi_2(\theta_2)\, d\theta_2}}$$

### Jeffreys Scale

| ln B₁₂ | Interpretation |
|--------|----------------|
| < 1.0 | Negligible evidence for M₁ |
| 1.0 – 2.5 | Substantial |
| 2.5 – 5.0 | Strong |
| > 5.0 | Decisive |

---

## 6.3 Occam's Razor is Automatic

The evidence penalises complexity without hand-tuning. Under a Laplace approximation:

$$\ln Z \approx \ln \mathcal{L}(\hat{\theta}) - \ln\frac{|\Sigma_{\rm prior}|^{1/2}}{|\Sigma_{\rm post}|^{1/2}}$$

A parameter the data cannot constrain contributes no compression → Occam factor grows → evidence falls. You don't need explicit penalty terms.

---

## 6.4 Information Criteria

When full evidence is too expensive:

**AIC:** `-2 ln L_max + 2k`  
**BIC:** `-2 ln L_max + k ln N` (heavier penalty for large N)  
**DIC:** `D_bar + p_D` where `p_D = D_bar - D(theta_bar)` (uses full posterior, good for hierarchical)  
**WAIC:** Most reliable for predictive accuracy, computed with ArviZ

---

## 6.5 Posterior Predictive Checks

1. Draw theta^(s) from posterior
2. Simulate D_rep^(s) from the likelihood
3. Compare test statistic T(D_rep) to T(D_obs)

If p_B = P(T(D_rep) ≥ T(D_obs) | D) ≈ 0 or 1, the model misses something.

---

## 6.6 Application: Spectral Line Detection

Fit M0 (continuum) and M1 (continuum + Gaussian line) to an X-ray spectrum using Poisson likelihood. Compute evidences via dynesty. The Bayes factor ln B_10 tells us decisively whether the line is real.

**Key insight:** A frequentist 3-sigma detection (p=0.003) often corresponds to ln B ≈ 2–3 — substantial but not decisive — especially for trials over many energy bins.

---

## 6.7 Prior Sensitivity

For model comparison, the prior width on extra parameters matters. Too broad a prior → more "wasted" prior volume → evidence penalty even if data support the feature. Always use physically motivated, bounded priors.

---

## 📝 Exercises

1. Fit a line vs quadratic to 20 noisy points. Compare AICc and ln B from nested sampling.
2. Simulate Poisson spectra with/without a line. Find the amplitude at which ln B > 5.
3. Run posterior predictive checks on a galaxy rotation curve fit.

---

## 🔗 Navigation
[← Chapter 5: Nested Sampling](../ch05_nested_sampling/README.md)  
[→ Chapter 7: Gaussian Processes](../ch07_gaussian_processes/README.md)
