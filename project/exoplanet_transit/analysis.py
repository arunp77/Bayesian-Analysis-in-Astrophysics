"""
project/exoplanet_transit/analysis.py
======================================
Capstone project: Bayesian exoplanet transit light curve analysis.

Target: WASP-39b (TESS data)
 - Period  ~ 4.055 days
 - Rp/R*   ~ 0.143
 - Impact b ~ 0.43

Phases:
  1. Data loading and preprocessing
  2. BLS period finding
  3. MCMC fit with emcee (transit parameters)
  4. GP + transit joint fit with celerite2
  5. Model comparison with dynesty
  6. Physical parameter derivation
"""

import numpy as np
import matplotlib.pyplot as plt
import batman
import emcee
import dynesty
import celerite2
from celerite2 import terms
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# ─── 1. Data Loading ──────────────────────────────────────────────────────────

def load_tess_lightcurve(target="WASP-39", sector=None):
    """
    Download and preprocess a TESS light curve using lightkurve.

    Returns
    -------
    t : array
        Time in days (BJD - reference)
    y : array
        Normalised flux
    yerr : array
        Flux uncertainty
    """
    try:
        import lightkurve as lk
        result = lk.search_lightcurve(target, mission="TESS", sector=sector)
        print(f"Found {len(result)} light curves for {target}")
        lc = result[0].download()
        lc = lc.normalize().remove_outliers(sigma=5)
        t = lc.time.value
        y = lc.flux.value
        yerr = lc.flux_err.value
        # Remove NaNs
        mask = np.isfinite(t) & np.isfinite(y) & np.isfinite(yerr)
        return t[mask], y[mask], yerr[mask]
    except ImportError:
        print("lightkurve not installed. Generating synthetic data instead.")
        return generate_synthetic_data()


def generate_synthetic_data(
    period=4.055, t0=0.5, rp=0.143, b=0.43,
    n_points=2000, noise_level=5e-4, seed=42
):
    """
    Generate a synthetic WASP-39b-like transit light curve for testing.
    """
    np.random.seed(seed)
    t = np.sort(np.random.uniform(0, 2 * period, n_points))

    params = batman.TransitParams()
    params.t0     = t0
    params.per    = period
    params.rp     = rp
    params.a      = 11.4         # semi-major axis in R_star (from literature)
    params.inc    = np.degrees(np.arccos(b / 11.4))
    params.ecc    = 0.0
    params.w      = 90.0
    params.u      = [0.4, 0.3]   # limb darkening
    params.limb_dark = "quadratic"

    m = batman.TransitModel(params, t)
    flux_model = m.light_curve(params)

    # Add correlated noise (red noise) + white noise
    noise = noise_level * np.random.randn(n_points)
    flux = flux_model + noise
    flux_err = noise_level * np.ones(n_points)

    return t, flux, flux_err


# ─── 2. Transit Model ─────────────────────────────────────────────────────────

def transit_model(theta, t):
    """
    Evaluate the Mandel-Agol transit model.

    Parameters
    ----------
    theta : array [t0, per, rp, a, inc, u1, u2]
    t : array of times

    Returns
    -------
    flux : array
    """
    t0, per, rp, a, inc, u1, u2 = theta

    params = batman.TransitParams()
    params.t0        = t0
    params.per       = per
    params.rp        = rp
    params.a         = a
    params.inc       = inc
    params.ecc       = 0.0
    params.w         = 90.0
    params.u         = [u1, u2]
    params.limb_dark = "quadratic"

    m = batman.TransitModel(params, t)
    return m.light_curve(params)


# ─── 3. Log-Posterior for MCMC ────────────────────────────────────────────────

def log_prior(theta):
    """
    Log prior for transit parameters.
    theta = [t0, per, rp, a, inc, u1, u2]
    """
    t0, per, rp, a, inc, u1, u2 = theta

    # Uniform priors (physical bounds)
    if not (0.0   < t0  < 10.0):   return -np.inf
    if not (3.0   < per < 5.5):    return -np.inf   # tight around known value
    if not (0.01  < rp  < 0.25):   return -np.inf
    if not (5.0   < a   < 20.0):   return -np.inf
    if not (70.0  < inc < 90.0):   return -np.inf
    if not (0.0   < u1  < 1.0):    return -np.inf
    if not (0.0   < u2  < 1.0):    return -np.inf
    if u1 + u2 > 1.0:              return -np.inf   # physical constraint

    # Gaussian prior on limb darkening from stellar models
    lp  = stats.norm.logpdf(u1, 0.40, 0.10)
    lp += stats.norm.logpdf(u2, 0.30, 0.10)
    return lp


def log_likelihood(theta, t, y, yerr):
    """
    Gaussian log-likelihood for transit model.
    """
    try:
        model = transit_model(theta, t)
    except Exception:
        return -np.inf
    return -0.5 * np.sum(((y - model) / yerr)**2 + np.log(2 * np.pi * yerr**2))


def log_posterior(theta, t, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, t, y, yerr)


# ─── 4. MCMC Fit ──────────────────────────────────────────────────────────────

def run_mcmc(t, y, yerr, theta_init, nwalkers=64, nburn=500, nprod=3000):
    """
    Run emcee MCMC for transit parameter estimation.

    Returns
    -------
    sampler : emcee.EnsembleSampler (after production run)
    flat_samples : array, shape (nwalkers * nprod, ndim)
    """
    ndim = len(theta_init)
    p0 = theta_init + 1e-4 * np.random.randn(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_posterior,
        args=(t, y, yerr)
    )

    print("Running burn-in...")
    sampler.run_mcmc(p0, nburn, progress=True)
    sampler.reset()

    print("Running production chain...")
    sampler.run_mcmc(None, nprod, progress=True)

    # Convergence diagnostics
    tau = sampler.get_autocorr_time(quiet=True)
    print(f"\nAutocorrelation times: {np.round(tau, 1)}")
    print(f"Effective samples: {np.round(nprod * nwalkers / tau, 0)}")

    flat_samples = sampler.get_chain(flat=True)
    return sampler, flat_samples


# ─── 5. GP + Transit Joint Fit ────────────────────────────────────────────────

def build_gp(gp_params, t, yerr):
    """
    Build a celerite2 GP with Matérn-3/2 kernel.

    gp_params = [log_sigma, log_rho]  (amplitude, length scale)
    """
    log_sigma, log_rho = gp_params
    kernel = terms.Matern32Term(sigma=np.exp(log_sigma), rho=np.exp(log_rho))
    gp = celerite2.GaussianProcess(kernel, mean=0.0)
    gp.compute(t, yerr=yerr)
    return gp


def log_posterior_gp(theta_full, t, y, yerr):
    """
    Log-posterior for joint transit + GP model.

    theta_full = [t0, per, rp, a, inc, u1, u2, log_sigma_gp, log_rho_gp]
    """
    theta_transit = theta_full[:7]
    gp_params     = theta_full[7:]

    lp = log_prior(theta_transit)
    if not np.isfinite(lp):
        return -np.inf

    # GP amplitude and length scale priors (log-uniform)
    log_sigma, log_rho = gp_params
    if not (-12 < log_sigma < -1):   return -np.inf
    if not (-4  < log_rho   < 4):    return -np.inf

    try:
        model = transit_model(theta_transit, t)
        residuals = y - model
        gp = build_gp(gp_params, t, yerr)
        ll = gp.log_likelihood(residuals)
    except Exception:
        return -np.inf

    return lp + ll


# ─── 6. Model Comparison via Nested Sampling ──────────────────────────────────

def run_nested_sampling(log_like_fn, prior_transform_fn, ndim, label=""):
    """
    Run dynesty nested sampling and return ln(Z).
    """
    print(f"\nRunning nested sampling for {label}...")
    sampler = dynesty.NestedSampler(
        log_like_fn, prior_transform_fn, ndim,
        nlive=400, bound='multi'
    )
    sampler.run_nested(dlogz=0.1, print_progress=True)
    results = sampler.results
    lnZ = results.logz[-1]
    lnZ_err = results.logzerr[-1]
    print(f"  ln Z = {lnZ:.2f} ± {lnZ_err:.2f}")
    return results, lnZ, lnZ_err


# ─── 7. Physical Parameter Derivation ─────────────────────────────────────────

def derive_physical_params(samples, R_star_Rsun=1.279, M_star_Msun=0.947):
    """
    Derive physical parameters from posterior samples.

    Parameters
    ----------
    samples : array, shape (N, 7)
        Columns: [t0, per, rp, a, inc, u1, u2]
    R_star_Rsun : float
        Stellar radius in solar radii (from spectroscopy)
    M_star_Msun : float
        Stellar mass in solar masses

    Returns
    -------
    dict of physical parameter arrays
    """
    R_sun = 6.957e8   # m
    M_sun = 1.989e30  # kg
    G     = 6.674e-11 # m^3 kg^-1 s^-2
    AU    = 1.496e11  # m
    R_Jup = 7.149e7   # m

    rp_over_rstar = samples[:, 2]
    a_over_rstar  = samples[:, 3]
    period_days   = samples[:, 1]

    R_star_m = R_star_Rsun * R_sun
    R_planet_m = rp_over_rstar * R_star_m
    a_m = a_over_rstar * R_star_m

    period_s = period_days * 86400.0
    T_eq = 5400 * np.sqrt(R_star_Rsun / (2 * a_m / AU))   # Kelvin (rough)

    return {
        "Rp_Rjup":   R_planet_m / R_Jup,
        "a_AU":      a_m / AU,
        "T_eq_K":    T_eq,
        "period_days": period_days,
    }


# ─── 8. Plotting ──────────────────────────────────────────────────────────────

def plot_phase_folded(t, y, yerr, theta_best, period, t0, title="Transit"):
    """Plot the phase-folded light curve with the best-fit model."""
    phase = ((t - t0) % period) / period
    phase[phase > 0.5] -= 1.0

    t_model = np.linspace(-0.15, 0.15, 1000) * period + t0
    f_model = transit_model(theta_best, t_model)
    phase_model = ((t_model - t0) % period) / period
    phase_model[phase_model > 0.5] -= 1.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})

    ax1.errorbar(phase, y, yerr=yerr, fmt='.', color='#888780',
                 alpha=0.4, ms=2, elinewidth=0.5, label='TESS data')
    sort_idx = np.argsort(phase_model)
    ax1.plot(phase_model[sort_idx], f_model[sort_idx],
             color='#3B8BD4', lw=2, label='Best-fit transit')
    ax1.set_ylabel("Normalised flux")
    ax1.legend(frameon=False)
    ax1.set_title(title)

    residuals = y - transit_model(theta_best, t)
    ax2.errorbar(phase, residuals, yerr=yerr, fmt='.', color='#888780',
                 alpha=0.4, ms=2, elinewidth=0.5)
    ax2.axhline(0, color='#3B8BD4', lw=1)
    ax2.set_xlabel("Orbital phase")
    ax2.set_ylabel("Residuals")

    plt.tight_layout()
    return fig


# ─── Main Analysis Pipeline ───────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("  WASP-39b Transit Analysis — Bayesian Pipeline")
    print("=" * 60)

    # 1. Load data
    print("\n[1] Loading TESS light curve...")
    t, y, yerr = load_tess_lightcurve("WASP-39")

    # 2. Initial parameter guess (from literature)
    theta_init = np.array([
        0.5,       # t0 (days, relative)
        4.0553,    # period (days)
        0.1430,    # rp = Rp/R*
        11.40,     # a = a/R* (semi-major axis in stellar radii)
        87.83,     # inclination (degrees)
        0.40,      # u1 limb darkening
        0.28,      # u2 limb darkening
    ])

    print(f"  Data points: {len(t)}")
    print(f"  Time baseline: {t[-1]-t[0]:.1f} days")
    print(f"  Median flux error: {np.median(yerr)*1e6:.1f} ppm")

    # 3. MCMC fit
    print("\n[2] Running MCMC parameter estimation...")
    sampler, samples = run_mcmc(t, y, yerr, theta_init,
                                nwalkers=64, nburn=500, nprod=3000)

    # 4. Results summary
    param_names = ["t0", "per", "rp", "a", "inc", "u1", "u2"]
    print("\n[3] Posterior Summary:")
    for i, name in enumerate(param_names):
        q = np.percentile(samples[:, i], [16, 50, 84])
        print(f"  {name:<6} = {q[1]:.5f} +{q[2]-q[1]:.5f} -{q[1]-q[0]:.5f}")

    # 5. Physical parameters
    phys = derive_physical_params(samples)
    print("\n[4] Physical Parameters:")
    for k, v in phys.items():
        q = np.percentile(v, [16, 50, 84])
        print(f"  {k:<15} = {q[1]:.4f} +{q[2]-q[1]:.4f} -{q[1]-q[0]:.4f}")

    # 6. Plot
    print("\n[5] Generating phase-folded plot...")
    theta_best = np.median(samples, axis=0)
    period_med = theta_best[1]
    t0_med     = theta_best[0]
    fig = plot_phase_folded(t, y, yerr, theta_best, period_med, t0_med,
                            title="WASP-39b — TESS Transit (Bayesian Fit)")
    plt.savefig("results/transit_fit.png", dpi=150, bbox_inches='tight')
    print("  Saved: results/transit_fit.png")

    print("\n[Done] Analysis complete.")
