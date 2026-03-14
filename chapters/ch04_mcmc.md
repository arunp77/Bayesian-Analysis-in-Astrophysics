# Chapter 4: Markov Chain Monte Carlo (MCMC)

> *"Monte Carlo is an extremely bad method; it should be used only when all alternative methods are worse."* — Alan Sokal

---

## 4.1 The Sampling Problem

For all but the simplest models, the posterior:
$$P(\theta \mid D) = \frac{\mathcal{L}(\theta) P(\theta)}{\int \mathcal{L}(\theta) P(\theta)\, d\theta}$$

has an **intractable normalisation integral**. With d parameters, brute-force numerical integration on an n-point grid scales as $O(n^d)$ — completely infeasible for d > 5.

**Solution:** Don't integrate. Instead, **draw samples** from $P(\theta \mid D)$ and use them to compute any desired quantity.

If we have samples $\{\theta^{(1)}, \theta^{(2)}, \ldots, \theta^{(S)}\}$ from $P(\theta \mid D)$:
$$E[f(\theta) \mid D] \approx \frac{1}{S} \sum_{s=1}^S f(\theta^{(s)})$$

By the Law of Large Numbers, this converges as $S^{-1/2}$ — independent of dimension!

---

## 4.2 Markov Chains

A **Markov chain** is a sequence of random variables $\theta^{(0)}, \theta^{(1)}, \theta^{(2)}, \ldots$ where each state depends only on the previous:
$$P(\theta^{(t+1)} \mid \theta^{(t)}, \theta^{(t-1)}, \ldots) = P(\theta^{(t+1)} \mid \theta^{(t)})$$

This is the **Markov property**.

A chain is **ergodic** if it eventually visits all regions of the state space with probability proportional to the target distribution π(θ). Under ergodicity:
$$\frac{1}{T}\sum_{t=1}^T f(\theta^{(t)}) \to E_\pi[f(\theta)] \quad \text{as } T \to \infty$$

---

## 4.3 The Metropolis-Hastings Algorithm

**Goal:** Sample from target distribution π(θ) ∝ P(θ | D).

**Algorithm:**
1. Initialise $\theta^{(0)}$ (randomly or at MLE)
2. At iteration t, propose $\theta^* \sim q(\theta^* \mid \theta^{(t)})$ (proposal distribution)
3. Compute acceptance ratio:
$$\boxed{r = \frac{\pi(\theta^*) \cdot q(\theta^{(t)} \mid \theta^*)}{\pi(\theta^{(t)}) \cdot q(\theta^* \mid \theta^{(t)})}}$$
4. Accept: $\theta^{(t+1)} = \theta^*$ with probability $\alpha = \min(1, r)$
5. Reject: $\theta^{(t+1)} = \theta^{(t)}$ with probability $1 - \alpha$

**Why does it work?** The acceptance probability enforces **detailed balance**:
$$\pi(\theta) \cdot T(\theta \to \theta') = \pi(\theta') \cdot T(\theta' \to \theta)$$

where T is the transition kernel. This guarantees π is the stationary distribution.

### 4.3.1 Symmetric Proposal: Metropolis Algorithm

If $q(\theta^* \mid \theta) = q(\theta \mid \theta^*)$ (e.g., Gaussian random walk: $\theta^* = \theta^{(t)} + \epsilon$, $\epsilon \sim \mathcal{N}(0, \Sigma)$):

$$r = \frac{\pi(\theta^*)}{\pi(\theta^{(t)})} = \frac{\mathcal{L}(\theta^*) P(\theta^*)}{\mathcal{L}(\theta^{(t)}) P(\theta^{(t)})}$$

In log space (numerically stable):
$$\ln r = \ln \mathcal{L}(\theta^*) + \ln P(\theta^*) - \ln \mathcal{L}(\theta^{(t)}) - \ln P(\theta^{(t)})$$

Accept if $\ln r > \ln U$ where $U \sim \text{Uniform}(0, 1)$.

### 4.3.2 Optimal Acceptance Rate

For a d-dimensional Gaussian target with Gaussian proposals:
- Optimal acceptance rate: **23.4%**
- Achieved by tuning the proposal covariance: $\Sigma_{\text{proposal}} \approx \frac{2.38^2}{d} \Sigma_{\text{posterior}}$

---

## 4.4 Gibbs Sampling

When the **full conditional distributions** $P(\theta_i \mid \theta_{-i}, D)$ are tractable:

**Algorithm:**
1. Initialise $(\theta_1^{(0)}, \theta_2^{(0)}, \ldots, \theta_d^{(0)})$
2. At step t, cycle through each parameter:
$$\theta_1^{(t+1)} \sim P(\theta_1 \mid \theta_2^{(t)}, \ldots, \theta_d^{(t)}, D)$$
$$\theta_2^{(t+1)} \sim P(\theta_2 \mid \theta_1^{(t+1)}, \theta_3^{(t)}, \ldots, \theta_d^{(t)}, D)$$
$$\vdots$$
$$\theta_d^{(t+1)} \sim P(\theta_d \mid \theta_1^{(t+1)}, \ldots, \theta_{d-1}^{(t+1)}, D)$$

Gibbs always accepts → more efficient when conditionals are available. Conjugate models (Chapter 2) yield tractable conditionals.

---

## 4.5 Ensemble Samplers: emcee

The **affine-invariant ensemble sampler** (Goodman & Weare 2010, implemented as `emcee`) uses an ensemble of **walkers** and is the most widely used MCMC in astrophysics.

**Stretch move** (the core proposal):

With two walkers at positions $\theta_k$ and $\theta_j$ (j ≠ k):
$$\theta_k^* = \theta_j + Z(\theta_k - \theta_j)$$

where $Z \sim g(z) \propto z^{-1/2}$ for $z \in [1/a, a]$ (default a = 2).

Acceptance probability:
$$r = Z^{d-1} \cdot \frac{\pi(\theta_k^*)}{\pi(\theta_k)}$$

**Advantages over Metropolis:**
- Automatically adapts to posterior shape
- Invariant under affine transformations (handles correlated parameters)
- Embarrassingly parallel across walkers
- Requires no tuning of proposal covariance

**Usage pattern in astrophysics:**
```python
import emcee

def log_posterior(theta, data):
    log_p = log_prior(theta)
    if not np.isfinite(log_p):
        return -np.inf
    return log_p + log_likelihood(theta, data)

nwalkers, ndim = 32, 5
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[data])

# Burn-in
sampler.run_mcmc(initial_positions, 1000, progress=True)
sampler.reset()

# Production
sampler.run_mcmc(None, 5000, progress=True)
flat_samples = sampler.get_chain(flat=True)
```

---

## 4.6 Hamiltonian Monte Carlo (HMC)

HMC treats θ as a **position** and introduces auxiliary **momentum** p, exploring the extended space of (θ, p) with Hamiltonian dynamics:

$$H(\theta, p) = -\ln \pi(\theta) + \frac{1}{2} p^T M^{-1} p$$

**Leapfrog integration:**
$$p_{t+\epsilon/2} = p_t + \frac{\epsilon}{2} \nabla_\theta \ln\pi(\theta_t)$$
$$\theta_{t+\epsilon} = \theta_t + \epsilon M^{-1} p_{t+\epsilon/2}$$
$$p_{t+\epsilon} = p_{t+\epsilon/2} + \frac{\epsilon}{2} \nabla_\theta \ln\pi(\theta_{t+\epsilon})$$

The gradient $\nabla_\theta \ln\pi(\theta)$ requires **differentiable** likelihood functions.

**NUTS (No-U-Turn Sampler):** Automatically determines step count — the basis of Stan and PyMC.

**Advantage:** Dramatically faster mixing in high dimensions (d > 10) compared to random-walk MCMC.

---

## 4.7 Convergence Diagnostics

**How do we know when the chain has converged?**

### 4.7.1 Gelman-Rubin Statistic $\hat{R}$

Run M chains from overdispersed starting points. For parameter θ:

$$\hat{R} = \sqrt{\frac{\hat{V}}{W}}$$

where:
- $W$ = mean within-chain variance
- $\hat{V}$ = estimated marginal posterior variance (between + within)

$\hat{R} \to 1$ as chains converge. Rule of thumb: $\hat{R} < 1.01$ indicates convergence.

### 4.7.2 Effective Sample Size (ESS)

Chains are autocorrelated. The effective number of independent samples:

$$N_{\text{eff}} = \frac{N}{1 + 2\sum_{k=1}^{\infty} \rho_k}$$

where $\rho_k$ is the autocorrelation at lag k. Target: $N_{\text{eff}} > 200$ for mean estimates.

**Integrated autocorrelation time:**
$$\tau_{\text{int}} = 1 + 2\sum_{k=1}^\infty \rho_k, \qquad N_{\text{eff}} = N/\tau_{\text{int}}$$

### 4.7.3 Trace Plots

Visual inspection of chain evolution — should look like "fuzzy caterpillars".

---

## 4.8 Burn-in and Thinning

**Burn-in:** The initial transient period before the chain reaches stationarity. Discard typically first 20-50% of samples.

**Thinning:** Keep every k-th sample to reduce autocorrelation. Rarely helpful — better to run longer and keep all samples.

---

## 4.9 Practical MCMC Workflow

```
1. Write log_posterior = log_likelihood + log_prior
2. Test with synthetic data (known truth)
3. Run short chain → check acceptance rate (tune proposal)
4. Run burn-in → check trace plots
5. Run production → compute R̂ and N_eff
6. Extract samples → compute summaries, corner plots
7. Posterior predictive check → does the model reproduce data?
```

---

## 📝 Exercises

1. **Implement MH from scratch:** Code a Metropolis sampler for N(0,1) using a Gaussian proposal. Plot acceptance rate vs proposal width. Find the optimal width.

2. **emcee warmup:** Fit a straight line y = mx + b to 20 noisy data points using emcee (nwalkers=32). Make a corner plot of the posterior on (m, b, σ_intrinsic).

3. **Convergence check:** Run two chains for Exercise 1 from very different starting points. Compute R̂ as a function of number of steps.

4. **Autocorrelation:** For your MH chain from Exercise 1, compute ρ_k for k = 1...50 and estimate τ_int and N_eff. How many raw samples ≈ 1000 effective samples?

---

## 🔗 Navigation

[← Chapter 3: Likelihood & Priors](../ch03_likelihood_priors/README.md)  
[→ Chapter 5: Nested Sampling](../ch05_nested_sampling/README.md)
