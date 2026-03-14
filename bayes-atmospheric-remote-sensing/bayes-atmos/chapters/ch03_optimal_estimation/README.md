# Chapter 3: Optimal Estimation — Analytical Bayesian Retrieval

Optimal Estimation (OE), introduced by Rodgers (1976, 2000), is the
**analytical Bayesian solution** for Gaussian priors and Gaussian noise.
It is used in nearly every operational satellite retrieval algorithm:
ECMWF/ERA5 4D-Var, MERRA-2, IASI temperature sounding, TROPOMI ozone, etc.

---

## 3.1 The Optimal Estimation Solution

For a **linear forward model** F(x) = Kx and Gaussian distributions:

    Prior:      x ~ N(x_a, S_a)
    Likelihood: y ~ N(Kx, S_ε)
    Posterior:  x|y ~ N(x̂, S̃)

The posterior mean and covariance are **analytically exact**:

$$\boxed{\hat{x} = x_a + S_a K^T (K S_a K^T + S_\varepsilon)^{-1} (y - K x_a)}$$

$$\boxed{\tilde{S} = (K^T S_\varepsilon^{-1} K + S_a^{-1})^{-1}}$$

Alternative (numerically equivalent) forms:

    x̂ = x_a + (KᵀSε⁻¹K + Sa⁻¹)⁻¹ KᵀSε⁻¹ (y - Kx_a)
    x̂ = x_a + G (y - Kx_a)

where G = (KᵀSε⁻¹K + Sa⁻¹)⁻¹ KᵀSε⁻¹ is the **gain matrix**.

The first form requires inversion of an m×m matrix; the second requires n×n.
Use the first when m < n (fewer measurements than state elements).

---

## 3.2 Physical Interpretation

The OE solution can be understood as a **weighted average** of prior and measurement:

    x̂ = (1 - A) x_a + A x_true + noise_contribution

where A is the **averaging kernel matrix**.

- Where A ≈ I: retrieval dominated by the measurement
- Where A ≈ 0: retrieval dominated by the prior

The gain matrix G = S̃ KᵀSε⁻¹ can also be written:

    G = Sa Kᵀ (K Sa Kᵀ + Sε)⁻¹

This shows that G upweights measurements that:
- Have small noise (small Sε elements)
- Have high sensitivity to the state (large K elements)
- Align with the prior uncertainty (large Sa)

---

## 3.3 Nonlinear Optimal Estimation — Gauss-Newton Iteration

For a nonlinear F(x), linearise around the current estimate xᵢ:

$$F(x) \approx F(x_i) + K_i (x - x_i), \quad K_i = \left.\frac{\partial F}{\partial x}\right|_{x_i}$$

Update rule:

$$x_{i+1} = x_a + \tilde{S}_i K_i^T S_\varepsilon^{-1} [y - F(x_i) + K_i(x_i - x_a)]$$

$$\tilde{S}_i = (K_i^T S_\varepsilon^{-1} K_i + S_a^{-1})^{-1}$$

Convergence criterion (χ²-test on the update step):

$$d_i^2 = (x_{i+1} - x_i)^T \tilde{S}_i^{-1} (x_{i+1} - x_i) \ll n$$

When d_i² ≪ n, the change in x is small compared to the posterior uncertainty.

---

## 3.4 Retrieval Diagnostics

### χ² Test (Fit Quality)

After convergence at x̂, the normalised misfit:

$$\chi^2_y = [y - F(\hat{x})]^T S_\varepsilon^{-1} [y - F(\hat{x})] / m$$

Should be ≈ 1 for a good fit. χ²_y > 1 → underfitting (model too constrained
by prior, or forward model error). χ²_y < 1 → overfitting or overestimated noise.

### Degrees of Freedom for Signal

    d_s = trace(A) = trace(S̃_i Ki^T Sε⁻¹ Ki)

### Information Content

    H = ½ ln det(Sa S̃⁻¹) = ½ Σᵢ ln(1 + σa,i²/σ̃ᵢ²)

where σ̃ᵢ and σa,i are the posterior and prior standard deviations.

---

## 3.5 Optimal Estimation in Atmospheric Data Assimilation

4D-Var (used in ECMWF, MERRA-2) extends OE to include:
- A time dimension (observations at different times within a window)
- A nonlinear forecast model M linking state at time t₀ to time t
- The adjoint of both F and M for efficient gradient computation

The 4D-Var cost function:

$$J(x_0) = \frac{1}{2}(x_0 - x_b)^T B^{-1}(x_0 - x_b) + \frac{1}{2}\sum_{t=0}^T [y_t - H_t(M_{0\to t}(x_0))]^T R_t^{-1} [y_t - H_t(M_{0\to t}(x_0))]$$

where:
- x_b = background (model first guess)
- B = background error covariance
- y_t = observations at time t
- H_t = observation operator at time t
- R_t = observation error covariance
- M_{0→t} = model propagator from time 0 to t

This is Bayesian inference applied to the entire NWP analysis cycle.

---

## 3.6 Worked Example: Single-Layer Temperature Retrieval

**Setup:** 1-layer atmosphere, one spectral channel.
- True temperature: T = 250 K
- Prior: T_a = 240 K, σ_a = 10 K
- Radiance measurement: y = B(ν, 250 K) + noise, σ_ε = 0.1 K (in BT units)
- Linearised Jacobian: K = ∂B/∂T evaluated at T_a

**Posterior:**

    σ̃² = 1/(1/σ_a² + K²/σ_ε²)
    T̂ = σ̃² · (T_a/σ_a² + K y_BT/σ_ε²)

where y_BT = B⁻¹(y) is the brightness temperature.

The posterior mean is a **precision-weighted average** of prior and measurement.

---

## 📝 Exercises

1. For K = 2, Sε = 0.5, Sa = 4 (scalars), compute x̂, S̃, A, d_s.
   How does d_s change if you double Sa? If you halve Sε?

2. Implement the Gauss-Newton iteration for a 1D nonlinear forward model
   F(T) = B(ν, T) (Planck function). Retrieve T from y with a prior T_a = 240 K.

3. For an IASI sounding with K ∈ ℝ⁶⁰⁰×⁴⁰, Sa ∈ ℝ⁴⁰×⁴⁰, Sε ∈ ℝ⁶⁰⁰×⁶⁰⁰:
   Which matrix inversion form is more efficient? Why?

4. Generate a synthetic 5-level temperature profile retrieval. Add forward model
   error (perturb K by 10%). How does this affect the retrieved profile and error bars?

---

## 🔗 Navigation
[← Chapter 2: The Measurement Equation](../ch02_forward_model/README.md)
[→ Chapter 4: MCMC Retrievals for Non-Gaussian Problems](../ch04_mcmc_retrieval/README.md)
