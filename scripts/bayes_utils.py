"""
scripts/bayes_utils.py
======================
Reusable Bayesian inference utilities for the curriculum.
Covers: log-posteriors, common priors, MCMC helpers, diagnostics.
"""

import numpy as np
from scipy import stats


# ─── Prior functions ─────────────────────────────────────────────────────────

def log_prior_uniform(x, low, high):
    """Log of a uniform prior on [low, high]. Returns -inf outside."""
    if low < x < high:
        return -np.log(high - low)
    return -np.inf


def log_prior_log_uniform(x, low, high):
    """Log of a log-uniform (Jeffreys scale) prior on [low, high]."""
    if low < x < high:
        return -np.log(x) - np.log(np.log(high / low))
    return -np.inf


def log_prior_gaussian(x, mu, sigma):
    """Log of a Gaussian prior N(mu, sigma^2)."""
    return stats.norm.logpdf(x, loc=mu, scale=sigma)


def log_prior_half_gaussian(x, sigma):
    """Log of a half-Gaussian prior (x > 0)."""
    if x <= 0:
        return -np.inf
    return stats.halfnorm.logpdf(x, scale=sigma)


# ─── Likelihood functions ─────────────────────────────────────────────────────

def log_likelihood_gaussian(data, model, sigma):
    """
    Gaussian log-likelihood for array data with scalar or array sigma.

    L = -0.5 * sum[(d - m)^2 / sigma^2 + ln(2*pi*sigma^2)]
    """
    return np.sum(stats.norm.logpdf(data, loc=model, scale=sigma))


def log_likelihood_poisson(counts, rate):
    """
    Poisson log-likelihood. counts and rate are arrays of the same shape.

    L = sum[counts * ln(rate) - rate - ln(counts!)]
    """
    return np.sum(stats.poisson.logpmf(counts, mu=rate))


def log_likelihood_heteroscedastic(data, model, sigma_obs, sigma_jitter=0.0):
    """
    Gaussian likelihood with a jitter term added in quadrature.
    Useful when there's unknown systematic noise.

    sigma_eff^2 = sigma_obs^2 + sigma_jitter^2
    """
    sigma_eff = np.sqrt(sigma_obs**2 + sigma_jitter**2)
    return log_likelihood_gaussian(data, model, sigma_eff)


# ─── MCMC helpers ─────────────────────────────────────────────────────────────

def gelman_rubin(chains):
    """
    Compute the Gelman-Rubin R-hat statistic for convergence.

    Parameters
    ----------
    chains : array, shape (M, N)
        M chains, each of length N.

    Returns
    -------
    float
        R-hat. Values < 1.01 indicate convergence.
    """
    M, N = chains.shape
    chain_means = chains.mean(axis=1)
    grand_mean = chain_means.mean()

    B = N / (M - 1) * np.sum((chain_means - grand_mean)**2)     # between-chain
    W = chains.var(axis=1, ddof=1).mean()                        # within-chain
    V_hat = (N - 1) / N * W + (M + 1) / (M * N) * B
    return np.sqrt(V_hat / W)


def effective_sample_size(chain):
    """
    Compute the effective sample size (ESS) accounting for autocorrelation.

    Parameters
    ----------
    chain : array, shape (N,)
        A single MCMC chain.

    Returns
    -------
    float
        Effective sample size.
    """
    N = len(chain)
    chain_centered = chain - chain.mean()
    acf = np.correlate(chain_centered, chain_centered, mode='full')
    acf = acf[N-1:] / acf[N-1]  # normalise

    # Sum until first negative autocorrelation
    tau = 1.0
    for k in range(1, N // 2):
        if acf[k] < 0:
            break
        tau += 2 * acf[k]

    return N / tau


def summarise_posterior(samples, param_names=None, ci=0.68):
    """
    Print a summary table of posterior samples.

    Parameters
    ----------
    samples : array, shape (N, d)
    param_names : list of str
    ci : float, credible interval level (default 68%)
    """
    d = samples.shape[1] if samples.ndim > 1 else 1
    if samples.ndim == 1:
        samples = samples[:, None]
    if param_names is None:
        param_names = [f"theta_{i}" for i in range(d)]

    alpha = (1 - ci) / 2
    print(f"{'Parameter':<15} {'Median':>10} {f'-{ci*100:.0f}%':>10} {f'+{ci*100:.0f}%':>10} {'ESS':>8}")
    print("-" * 55)
    for i, name in enumerate(param_names):
        s = samples[:, i]
        med = np.median(s)
        lo = med - np.percentile(s, alpha * 100)
        hi = np.percentile(s, (1 - alpha) * 100) - med
        ess = effective_sample_size(s)
        print(f"{name:<15} {med:>10.4f} {lo:>+10.4f} {hi:>+10.4f} {ess:>8.0f}")


# ─── Model comparison utilities ───────────────────────────────────────────────

def dic_from_samples(log_like_samples, theta_mean_log_like):
    """
    Compute DIC from MCMC samples.

    Parameters
    ----------
    log_like_samples : array, shape (N,)
        Log-likelihood evaluated at each posterior sample.
    theta_mean_log_like : float
        Log-likelihood at the posterior mean of theta.

    Returns
    -------
    float, float
        DIC, effective number of parameters p_D
    """
    D_bar = np.mean(-2 * log_like_samples)
    D_theta_bar = -2 * theta_mean_log_like
    p_D = D_bar - D_theta_bar
    DIC = D_bar + p_D
    return DIC, p_D


def compare_aic_bic(log_like_max, k, n):
    """
    Compute AIC and BIC.

    Parameters
    ----------
    log_like_max : float
        Maximum log-likelihood.
    k : int
        Number of free parameters.
    n : int
        Number of data points.

    Returns
    -------
    dict with AIC, AICc, BIC
    """
    aic = -2 * log_like_max + 2 * k
    aicc = aic + 2 * k * (k + 1) / (n - k - 1) if n > k + 1 else np.inf
    bic = -2 * log_like_max + k * np.log(n)
    return {"AIC": aic, "AICc": aicc, "BIC": bic}


# ─── Example usage ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import emcee

    # Simple line-fitting example: y = m*x + b + noise
    np.random.seed(42)
    x = np.linspace(0, 10, 30)
    y_true = 2.5 * x + 1.0
    y_obs = y_true + np.random.normal(0, 1.5, size=len(x))
    sigma = 1.5 * np.ones_like(x)

    # ─── Posterior function ─────────────────────────────────────────────────────

    # m is slope of the line, b is the intercept, log_s is the log of the jitter i.e noise
    # We use log_s to ensure that s is always positive and log_s is uniform in [-3, 3]
    # This is equivalent to a log-uniform prior on s in [exp(-3), exp(3)] which is approx [0.05, 20] 
    def log_posterior(theta):
        m, b, log_s = theta
        lp = (log_prior_uniform(m, -10, 10) +
              log_prior_uniform(b, -20, 20) +
              log_prior_uniform(log_s, -3, 3))
        if not np.isfinite(lp):
            return -np.inf
        s = np.exp(log_s)
        # log_likelihood_heteroscedastic(data, model, sigma_obs, sigma_jitter=0.0) 
        # data = y_obs, model = m * x + b, sigma_obs = sigma, sigma_jitter = s 
        ll = log_likelihood_heteroscedastic(y_obs, m * x + b, sigma, sigma_jitter=s)
        return lp + ll

    # Number of walkers and dimensions 
    nwalkers, ndim = 32, 3

    # Initial positions of the walkers (randomly distributed around the true values) 
    # True values are m = 2.5, b = 1.0, log_s = 0.0 (starting values for m, b and log_s)
    # 0.1 is the spread around the true values 
    # np.random.randn(nwalkers, ndim) generates a matrix of random numbers with shape (nwalkers, ndim) 
    # and this helps avoid all walkers starting at same place 
    p0 = np.array([2.5, 1.0, 0.0]) + 0.1 * np.random.randn(nwalkers, ndim)

    # ─── MCMC sampling ───────────────────────────────────────────────────────────
    # nwalkers: number of chains
    # ndim: number of parameters
    # log_posterior: function that returns the log-posterior
    # p0: initial guess for the parameters
    # 2000: number of steps for the burn-in
    # 3000: number of steps for the production run
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)

    # Run the MCMC sampler
    # p0: initial guess for the parameters
    # 2000: number of steps for the burn-in
    # 3000: number of steps for the production run
    sampler.run_mcmc(p0, 2000, progress=True)
    sampler.reset()
    sampler.run_mcmc(None, 3000, progress=True)

    flat_samples = sampler.get_chain(flat=True)
    summarise_posterior(flat_samples, param_names=["m", "b", "log_sigma"])
