## Aerosol Optical Depth (AOD) — The Key Aerosol Parameter

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
