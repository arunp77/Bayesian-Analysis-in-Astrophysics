# Chapter 1: Radiative Transfer — The Physical Forward Model

## 1.1 Why Radiative Transfer Underpins Everything

Every satellite, every lidar, every radiometer measures **electromagnetic radiation**
that has interacted with the atmosphere. Understanding how radiation propagates
through a gaseous, scattering, absorbing medium is the first step to inverting
any atmospheric measurement.

The connection to Bayesian inference is direct:

    observation  y  =  F(x) + ε
    
    x  = atmospheric state vector (temperature profile, gas amounts, aerosols...)
    y  = measured radiances / reflectances / brightness temperatures
    F  = radiative transfer forward model
    ε  ~ N(0, S_ε)  =  instrument noise

The likelihood P(y | x) = P(ε = y − F(x)). Everything else in the retrieval
follows from this equation plus a prior P(x).

---

## 1.2 Beer-Lambert Law — Absorption in a Single Layer

For a monochromatic beam of intensity I₀ through an absorbing path of length L:

$$I(L) = I_0 \exp(-\tau)$$

where the **optical depth** τ = k_abs · L (dimensionless).

For a vertically inhomogeneous atmosphere:

$$\tau(\nu) = \int_0^\infty k(\nu, p, T)\, \rho(z)\, dz$$

The **transmittance** T(ν) = exp(−τ(ν)) ∈ [0, 1].

Key numbers:
- τ ≪ 1 → atmosphere is transparent (window channel)
- τ ≈ 1 → half the radiation absorbed (weighting function peaks here)
- τ ≫ 1 → atmosphere is opaque (strong absorption band)

---

## 1.3 The Planck Function — Thermal Emission

Every object at temperature T emits radiation. At wavenumber ν (cm⁻¹):

$$B(\nu, T) = \frac{2hc^2\nu^3}{\exp(hc\nu / kT) - 1}$$

Constants: h = 6.626×10⁻³⁴ J·s, c = 3×10¹⁰ cm/s, k = 1.38×10⁻²³ J/K

Peak wavenumber (Wien's law): ν_peak ≈ 1.96 kT/hc

| Temperature | Physical level | ν_peak (cm⁻¹) | λ_peak (μm) |
|---|---|---|---|
| 200 K | Stratosphere | 417 | 24 |
| 250 K | Upper troposphere | 520 | 19 |
| 300 K | Surface | 626 | 16 |

---

## 1.4 Top-of-Atmosphere Radiance (Thermal IR)

A nadir-viewing satellite in the thermal infrared (IASI, AIRS, CrIS, SEVIRI)
observes:

$$I_\text{TOA}(\nu) = \underbrace{\varepsilon_s B(\nu, T_s)\, t(\nu, p_s)}_\text{surface emission} + \int_{p_s}^{0} B(\nu, T(p))\, \frac{\partial t(\nu,p)}{\partial p}\, dp$$

where:
- $t(ν, p)$ = exp(−τ(ν, p)) = transmittance from pressure level p to space
- $T_s$ = surface skin temperature
- $ε_s$ = surface emissivity
- $T(p)$ = atmospheric temperature at pressure level p

The integral term is the **atmospheric emission contribution**. Each pressure
level contributes B(ν, T(p)) weighted by how much radiation escapes to space.

---

## 1.5 Weighting Functions (Jacobians) — Where Does the Signal Come From?

The **weighting function** (or Jacobian for temperature) is:

$$K_T(\nu, p) = \frac{\partial I_\text{TOA}(\nu)}{\partial T(p)} = \frac{\partial B(\nu, T(p))}{\partial T}\, \frac{\partial t(\nu, p)}{\partial p}$$

This tells us: *"if temperature at pressure level p changes by 1 K, how much does
the TOA radiance at channel ν change?"*

Key insight: $K_T(\nu, p)$ peaks where $\partial t/\partial p$ is large — i.e., where $\tau \approx 1$.

**The vertical sounding principle:** By measuring at many spectral channels with
different absorption strengths, we probe different altitude layers:
- Window channel ($\nu = 900$ cm⁻¹): peaks near surface ($\tau \approx 0$ from above)
- CO₂ band flank ($\nu = 680$ cm⁻¹): peaks in mid-troposphere
- CO₂ band centre (ν = 667 cm⁻¹): peaks in stratosphere

This is exactly why AIRS has 2378 channels — different channels give independent
information about different atmospheric levels.

---

## 1.6 Solar Reflectance Measurements (UV–Vis–NIR)

For shortwave sensors (MODIS, TROPOMI, Sentinel-2, Sentinel-5P) measuring
reflected sunlight:

$$R_\text{TOA} = R_\text{atm}(\theta_s, \theta_v, \phi) + \frac{T_\downarrow \cdot T_\uparrow \cdot R_s}{1 - S \cdot R_s}$$

where:
- R_atm = path reflectance from atmospheric scattering (Rayleigh + aerosol)
- T↓, T↑ = downwelling and upwelling transmittances
- R_s = surface reflectance
- S = atmospheric spherical albedo (accounts for multiple reflections)
- θ_s, θ_v, φ = solar zenith, view zenith, relative azimuth angles

This is the **surface-atmosphere decoupling** problem: measuring R_TOA, we want
to separate the atmospheric contribution (dominated by aerosols) from the surface.

---

## 1.7 Aerosol Optical Depth (AOD) — The Key Aerosol Parameter

AOD at wavelength λ integrates extinction over the atmospheric column:

$$\text{AOD}(\lambda) = \int_0^\infty k_{\text{ext,aer}}(\lambda, z)\, dz$$

The **Ångström exponent** α parameterises the spectral dependence:

$$\text{AOD}(\lambda) = \text{AOD}(\lambda_0) \cdot \left(\frac{\lambda}{\lambda_0}\right)^{-\alpha}$$

Physical interpretation:
- α ≈ 0: coarse particles (mineral dust, sea salt) — spectrally flat
- α ≈ 1–2: fine particles (biomass burning smoke, urban pollution)
- α > 2: very small particles (fresh combustion)

**MODIS retrieval** uses reflectances at 3 solar bands (470, 550, 2130 nm)
to simultaneously retrieve AOD and surface reflectance.

---

## 1.8 The Forward Model in Matrix Form

For a discrete state vector x ∈ ℝⁿ and measurement vector y ∈ ℝᵐ:

    y = F(x) + ε,   ε ~ N(0, Sε)

If F is differentiable, the **Jacobian matrix** K:

$$K_{ij} = \frac{\partial F_i(x)}{\partial x_j}$$

tells us how each state variable affects each measurement.

For a linearised problem near a reference state x_0:

    F(x) ≈ F(x_0) + K(x - x_0)

    → y - F(x_0) ≈ K · (x - x_0) + ε

This is the **linearised inverse problem** that Optimal Estimation solves analytically.

---

## 📝 Exercises

1. Plot B(ν, T) for T = 200, 250, 300 K over 200–2500 cm⁻¹. Verify Wien's law.

2. A CO₂ absorption channel has k_abs = 5×10⁻³ m² kg⁻¹ and CO₂ column
   amount = 8 kg m⁻². Compute τ and the transmittance. Is this channel in
   the transparent, moderately absorbing, or opaque regime?

3. MODIS measures AOD = 0.6 at 470 nm and AOD = 0.3 at 860 nm.
   Compute the Ångström exponent. What aerosol type does this suggest?

4. Why does the weighting function peak at τ ≈ 1? Give a physical argument
   (what happens for τ ≪ 1 and τ ≫ 1?)

---

## 🔗 Navigation
[→ Chapter 2: The Measurement Equation and Bayesian Framework](../ch02_forward_model/README.md)
