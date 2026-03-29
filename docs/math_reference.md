# Mathematical Reference

## Quick Formula Sheet

### Core Bayes

$$P(\theta \mid D) = \frac{P(D \mid \theta)\, P(\theta)}{P(D)}$$

### Gaussian PDF

$$\mathcal{N}(x; \mu, \sigma^2) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

### Poisson PMF

$$P(k; \lambda) = \frac{\lambda^k e^{-\lambda}}{k!}$$

### Log-Likelihood (Gaussian noise)

$$\ln \mathcal{L}(\theta) = -\frac{1}{2}\sum_i \left[\frac{(d_i - m_i(\theta))^2}{\sigma_i^2} + \ln(2\pi\sigma_i^2)\right]$$

### Metropolis-Hastings Acceptance

$$\alpha = \min\!\left(1,\; \frac{\pi(\theta^*)\, q(\theta^{(t)} \mid \theta^*)}{\pi(\theta^{(t)})\, q(\theta^* \mid \theta^{(t)})}\right)$$

### GP Posterior Mean and Variance

$$\mu_* = \mathbf{k}_*^T (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{y}$$
$$\sigma_*^2 = k_{**} - \mathbf{k}_*^T (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{k}_*$$

### Nested Sampling Evidence

$$\ln Z \approx \ln\sum_i w_i \mathcal{L}_i, \quad w_i \approx e^{-i/N}\!\left(1 - e^{-1/N}\right)$$

### Transit Depth

$$\delta = \left(\frac{R_p}{R_*}\right)^2$$

### Kepler's Third Law

$$a^3 = \frac{G M_* P^2}{4\pi^2}$$

---

## Glossary

| Term                   | Definition                                               |
| ---------------------- | -------------------------------------------------------- |
| **Posterior**          | P(θ\|D) — probability of parameters given data           |
| **Prior**              | P(θ) — belief about parameters before seeing data        |
| **Likelihood**         | P(D\|θ) — probability of data given parameters           |
| **Evidence**           | P(D) — normalisation constant; used for model comparison |
| **Conjugate prior**    | Prior that keeps posterior in same family as prior       |
| **MAP estimate**       | Maximum A Posteriori — mode of posterior                 |
| **Credible interval**  | Bayesian analogue of confidence interval                 |
| **HPD interval**       | Highest Posterior Density — shortest credible interval   |
| **Marginalisation**    | Integrating out nuisance parameters                      |
| **MCMC**               | Markov Chain Monte Carlo — sampling algorithm            |
| **Burn-in**            | Initial transient period of MCMC chain (discarded)       |
| **ESS**                | Effective Sample Size — accounts for autocorrelation     |
| **R̂**                  | Gelman-Rubin statistic — convergence diagnostic          |
| **Bayes factor**       | Ratio of evidences — for model comparison                |
| **GP**                 | Gaussian Process — distribution over functions           |
| **Kernel**             | Covariance function defining GP smoothness               |
| **Hyperparameters**    | Parameters of the GP kernel                              |
| **Hierarchical model** | Multi-level model with population hyperpriors            |
| **Shrinkage**          | Hierarchical partial pooling toward population mean      |
| **Fisher information** | Expected curvature of log-likelihood                     |
| **Jeffreys prior**     | Invariant prior based on Fisher information              |
