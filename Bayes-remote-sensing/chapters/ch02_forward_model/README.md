# Chapter 2: The Measurement Equation and Bayesian Framework

## 2.1 From Radiance to State — The Inverse Problem

The forward model tells us how to go from atmosphere → measurement.
The retrieval algorithm goes the other way: measurement → atmosphere.

This is an **ill-posed inverse problem** because:

1. We have fewer measurements than unknown state variables (underdetermined)
2. Many different atmospheric states can produce the same spectrum (non-unique)
3. Small measurement errors can produce large errors in the retrieved state
   if the forward model is not well-conditioned (ill-conditioned)

Bayesian inference provides the principled solution through the posterior P(x|y).

---

## 2.2 The State Vector and Prior

The **atmospheric state vector** x contains everything we want to retrieve.
For a temperature sounding problem:

$$
\mathbf{x} =
\begin{bmatrix}
T(p_1),T(p_2), \ldots, T(p_N), T_s,\varepsilon_s
\end{bmatrix}^T \in \mathbb{R}^n
$$

For an aerosol retrieval:

$$
\mathbf{x} =
\begin{bmatrix}
\mathrm{AOD}_{550}, \alpha, \omega_0, R_s(470\,\mathrm{nm}), R_s(550\,\mathrm{nm}), R_s(860\,\mathrm{nm})
\end{bmatrix}^T \in \mathbb{R}^6
$$

The **prior distribution** encodes our knowledge before the measurement:

$$
\mathbf{x} \sim \mathcal{N}(\mathbf{x}_a,\mathbf{S}_a)
$$

where:

- $x_a$ = **a priori** state (from climatology, NWP model, or previous retrieval)
- $S_a$ = **a priori covariance matrix** (how uncertain we are about each element)

The a priori covariance $S_a$ encodes both:

- **Variances** on the diagonal: σᵢ² = uncertainty in the i-th state element
- **Correlations** off-diagonal: adjacent atmospheric levels are correlated

---

## 2.3 The Likelihood — Instrument Noise Model

Assuming Gaussian instrument noise:

    ε ~ N(0, Sε)

where $S_ε$ is the **measurement error covariance matrix**.

For independent channels: $S_ε = \mathrm{diag}(\sigma_1^2, \sigma_2^2, \ldots, \sigma_m^2)$

The log-likelihood is then (equation 57 of CO2 notes):

$$\ln P(y | x) = -\frac{1}{2} [y - F(x)]^T S_\varepsilon^{-1} [y - F(x)] + \text{const}$$

This is equivalent to a weighted χ² misfit between measured and forward-modelled radiances.

---

## 2.4 The Full Bayesian Posterior

Combining prior and likelihood:

$$P(x | y) \propto P(y | x) \cdot P(x)$$

$$\ln P(x | y) = -\frac{1}{2}\underbrace{[y - F(x)]^T S_\varepsilon^{-1} [y - F(x)]}_\text{misfit to data} -\frac{1}{2}\underbrace{(x - x_a)^T S_a^{-1} (x - x_a)}_\text{penalty from prior} + \text{const}$$

The retrieval minimises the **cost function** (equation 59 of the CO2 notes):

$$\boxed{J(x) = [y - F(x)]^T S_\varepsilon^{-1} [y - F(x)] + (x - x_a)^T S_a^{-1} (x - x_a)}$$

This is the **generalised least squares** cost function. The second term is a
**Tikhonov regularisation** — it prevents unphysical solutions by penalising
departures from the a priori.

---

## 2.5 The Jacobian Matrix K

For a forward model F: ℝⁿ → ℝᵐ, the Jacobian is (discussed in subsection 6.4.1):

$$K = \frac{\partial F}{\partial x} \in \mathbb{R}^{m \times n}$$

Each row Kᵢ tells us how measurement i responds to changes in all state elements.
Each column Kⱼ tells us how all measurements respond to changes in state element j.

**Computing K in practice:**

- Finite differences:

$$K_{ij} = \frac{[F_i(x + δe_j) - F_i(x)]}{δ}$$

(slow, O(n) forward model calls)

- **Analytic Jacobians:** derived from the physics (fast, error-prone to code)
- **Adjoint method:** compute $K^T v$ efficiently for any vector $v$ (used in variational data assimilation)

**Gain Matrix:**

$$G = (K^T S_\varepsilon^{-1} K + S_a^{-1})^{-1} K^T S_\varepsilon^{-1}$$

In terms of Gain matrix the retrieval is given by:

$$\hat{x} = \mathbf{x}_a + G(\mathbf{y} - F(\mathbf{x}_a))$$

Now linearize the forward model around:

$$F(x) \approx F(x_a) + K(x - x_a)$$

Then we can write the retrieval as:

$$\hat{x} - x_a = A(\hat{x} - x_a) + G\epsilon$$

where A is the averaging kernel matrix and G is the gain matrix.

---

## 2.6 Degrees of Freedom for Signal (DFS)

How much independent information does the measurement provide about the state? (equation 63 of CO2 notes)

    A = GK = (KᵀSε⁻¹K + Sa⁻¹)⁻¹ KᵀSε⁻¹K    (Averaging kernel matrix)

The **degrees of freedom for signal** (DFS):

    d_s = trace(A) ∈ [0, n]

where n = dimension of state vector.

- d_s ≈ 0: measurement tells us nothing new (prior dominates)
- d_s ≈ n: measurement fully determines all state elements (data dominates)
- Typical IASI temperature sounding: d_s ≈ 8–12 for a 40-level profile

The averaging kernel matrix A tells us what the retrieval really measures:

    x̂ - x_a = A(x_true - x_a) + noise

Each row Aᵢ is the "true" profile that would be retrieved as a delta perturbation
at level i — a broader Aᵢ row means coarser vertical resolution.

---

## 2.7 Information Content

The **Shannon information content** gained from the measurement:

$$H = \frac{1}{2} \ln \det(I + S_a^{1/2} K^T S_\varepsilon^{-1} K S_a^{1/2})$$

This measures how much the posterior is "narrower" than the prior (in bits or nats).
Used to optimally select measurement channels that maximise retrieval information.

---

## 2.8 Error Budget

The total retrieval error has contributions:

| Source            | Expression                              | Name                  |
| ----------------- | --------------------------------------- | --------------------- |
| Measurement noise | S_noise = (KᵀSε⁻¹K + Sa⁻¹)⁻¹ KᵀSε⁻¹K Sa | Noise error           |
| Smoothing         | S_smooth = (A - I) Sa (A - I)ᵀ          | Smoothing error       |
| Forward model     | S_FM = G Kb Sb Kbᵀ Gᵀ                   | Model parameter error |
| Total             | Sₓ = S_noise + S_smooth + S_FM          | Total error           |

where G = (KᵀSε⁻¹K + Sa⁻¹)⁻¹ KᵀSε⁻¹ is the gain matrix and Kb is
the Jacobian with respect to model parameters b (e.g., spectroscopic data).

---

## 📝 Exercises

1. For a 2×2 system, K = [[2,0],[0,3]], Sε = I, Sa = I.
   Compute the posterior covariance, averaging kernel matrix, and DFS.

2. What physical meaning does a diagonal averaging kernel (A ≈ I) have?
   What does it mean when A has broad off-diagonal elements?

3. In MODIS aerosol retrieval, why is it harder to retrieve AOD over bright
   desert surfaces than over dark ocean? Think in terms of the Jacobian K
   and the signal-to-noise ratio K · σ_x / σ_meas.

4. Design a prior S_a for a temperature profile retrieval at 20 pressure levels.
   What off-diagonal structure makes physical sense?

---

## 🔗 Navigation

[← Chapter 1: Radiative Transfer](../ch01_radiative_transfer/README.md)
[→ Chapter 3: Optimal Estimation](../ch03_optimal_estimation/README.md)

$$
$$
