# Chapter 1: Probability Foundations

> *"Probability is the very guide of life."* — Cicero

---

## 1.1 Why Probability in Astrophysics?

Astrophysics is, at its core, a science of inference. We cannot run controlled experiments on distant stars. We cannot rewind a supernova. We observe photons that have traveled billions of years and ask: *what physical process created them?*

Every measurement is corrupted by noise. Every model is an approximation. Every conclusion carries uncertainty. **Probability theory is the mathematical language for reasoning under uncertainty.**

---

## 1.2 Sample Spaces and Events

**Definition (Sample Space):** The sample space Ω is the set of all possible outcomes of a random experiment.

**Examples in astrophysics:**
- Ω = ℝ for a flux measurement
- Ω = ℝ² for a (RA, Dec) position
- Ω = {0, 1, 2, ...} for photon counts in a CCD pixel

An **event** A is any subset A ⊆ Ω.

---

## 1.3 The Kolmogorov Axioms

All of probability theory rests on three axioms (Kolmogorov, 1933):

**Axiom 1 (Non-negativity):**
$$P(A) \geq 0 \quad \forall A \subseteq \Omega$$

**Axiom 2 (Normalization):**
$$P(\Omega) = 1$$

**Axiom 3 (Countable Additivity):** For mutually exclusive events $A_1, A_2, \ldots$:
$$P\!\left(\bigcup_{i=1}^{\infty} A_i\right) = \sum_{i=1}^{\infty} P(A_i)$$

From these three axioms, *all* of probability theory follows deductively.

**Derived results:**
$$P(\emptyset) = 0$$
$$P(A^c) = 1 - P(A)$$
$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

---

## 1.4 Conditional Probability

The **conditional probability** of event A given that B has occurred:

$$\boxed{P(A \mid B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0}$$

**Physical interpretation:** Conditioning *restricts* the sample space from Ω to B.

**Example:** A radio telescope detects a burst. Given that the burst came from a known galaxy cluster (event B), what is the probability it is an FRB (event A)?

### 1.4.1 The Chain Rule

Applying conditional probability repeatedly:

$$P(A \cap B) = P(A \mid B) \cdot P(B)$$

For $n$ events:
$$P(A_1 \cap A_2 \cap \cdots \cap A_n) = P(A_1) \cdot P(A_2 \mid A_1) \cdot P(A_3 \mid A_1, A_2) \cdots$$

### 1.4.2 Independence

Events A and B are **independent** if:
$$P(A \cap B) = P(A) \cdot P(B) \iff P(A \mid B) = P(A)$$

Knowing B gives *no information* about A.

---

## 1.5 The Law of Total Probability

If $\{B_1, B_2, \ldots, B_n\}$ is a **partition** of Ω (mutually exclusive, exhaustive), then:

$$\boxed{P(A) = \sum_{i=1}^{n} P(A \mid B_i) \cdot P(B_i)}$$

This is the *marginalisation* formula in disguise — we will use it constantly.

**Astrophysical example:** A transient event A could be caused by several astrophysical processes: GRB, magnetar flare, or instrument artifact. If we know the prior probability of each cause $B_i$ and the probability of observing A under each cause, we can compute the total probability of A.

---

## 1.6 Random Variables

A **random variable** X is a function X: Ω → ℝ that maps outcomes to real numbers.

### 1.6.1 Discrete Random Variables

X takes countable values $\{x_1, x_2, \ldots\}$.

The **probability mass function (PMF)**:
$$p(x_k) = P(X = x_k)$$

Properties:
$$p(x_k) \geq 0, \qquad \sum_k p(x_k) = 1$$

**Key discrete distributions in astrophysics:**

#### Poisson Distribution
For photon counting, radioactive decay, cosmic ray hits:
$$\boxed{P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \ldots}$$

where λ > 0 is the expected count rate. Properties:
- Mean: E[X] = λ
- Variance: Var(X) = λ
- In the limit of large λ, approaches Gaussian

#### Binomial Distribution
For binary outcomes (detection/non-detection):
$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$

Mean: np, Variance: np(1-p)

### 1.6.2 Continuous Random Variables

X takes values in an interval. The **probability density function (PDF)** f(x) satisfies:
$$P(a \leq X \leq b) = \int_a^b f(x)\, dx$$

Properties:
$$f(x) \geq 0, \qquad \int_{-\infty}^{\infty} f(x)\, dx = 1$$

**Key continuous distributions in astrophysics:**

#### Gaussian (Normal) Distribution
The workhorse of measurement uncertainty:
$$\boxed{f(x \mid \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)}$$

Parameters: mean μ, standard deviation σ.  
Notation: X ~ N(μ, σ²)

#### Multivariate Gaussian
For correlated measurements (e.g., flux and color):
$$f(\mathbf{x} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{d/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\!\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$$

where Σ is the **covariance matrix**.

#### Cauchy / Lorentzian Distribution
Appears in spectral line profiles (natural line broadening):
$$f(x \mid x_0, \gamma) = \frac{1}{\pi\gamma\left[1 + \left(\frac{x-x_0}{\gamma}\right)^2\right]}$$

Note: No finite mean or variance — heavy tails matter in astronomy!

---

## 1.7 Expectation and Moments

The **expectation** (mean) of X:
$$E[X] = \int_{-\infty}^{\infty} x\, f(x)\, dx$$

The **variance**:
$$\text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2$$

The **n-th moment**:
$$\mu_n = E[X^n] = \int_{-\infty}^{\infty} x^n f(x)\, dx$$

**Moment Generating Function (MGF):**
$$M_X(t) = E[e^{tX}]$$

Taking derivatives: $\frac{d^n M_X}{dt^n}\bigg|_{t=0} = E[X^n]$

---

## 1.8 Joint, Marginal, and Conditional Distributions

For two continuous random variables X, Y with joint PDF f(x, y):

**Marginal PDF of X:**
$$\boxed{f_X(x) = \int_{-\infty}^{\infty} f(x, y)\, dy}$$

This is **marginalisation** — integrating out the nuisance variable Y.

**Conditional PDF:**
$$f(x \mid y) = \frac{f(x, y)}{f_Y(y)}$$

**Covariance:**
$$\text{Cov}(X, Y) = E[(X - \mu_X)(Y - \mu_Y)] = E[XY] - \mu_X \mu_Y$$

**Correlation coefficient:**
$$\rho_{XY} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y} \in [-1, 1]$$

---

## 1.9 The Central Limit Theorem

**Theorem:** Let $X_1, X_2, \ldots, X_n$ be i.i.d. with mean μ and variance σ². Then:

$$\boxed{\sqrt{n}\left(\bar{X}_n - \mu\right) \xrightarrow{d} \mathcal{N}(0, \sigma^2) \quad \text{as } n \to \infty}$$

where $\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i$.

**Why it matters in astrophysics:** Sums of many independent noise sources (read noise, dark current, shot noise) approach a Gaussian — this justifies Gaussian error propagation in many contexts, but not all! When counts are low (< ~20 photons), Poisson ≠ Gaussian.

---

## 1.10 Information Theory: Entropy and KL Divergence

**Shannon Entropy** measures the uncertainty of a distribution:
$$H[p] = -\int p(x) \ln p(x)\, dx$$

**Kullback-Leibler Divergence** measures how much distribution q differs from reference p:
$$D_{\text{KL}}(p \| q) = \int p(x) \ln\frac{p(x)}{q(x)}\, dx \geq 0$$

Equality holds iff p = q. This will reappear in Bayesian model comparison.

---

## 1.11 Summary

| Concept | Formula | Use in Astrophysics |
|---------|---------|---------------------|
| Conditional probability | $P(A\|B) = P(A\cap B)/P(B)$ | Updating beliefs on observations |
| Total probability | $P(A) = \sum_i P(A\|B_i)P(B_i)$ | Marginalising over hypotheses |
| Poisson distribution | $P(k) = \lambda^k e^{-\lambda}/k!$ | Photon counting, event rates |
| Gaussian distribution | $f(x) \propto \exp(-(x-\mu)^2/2\sigma^2)$ | Measurement noise model |
| Marginalisation | $f_X(x) = \int f(x,y)\,dy$ | Eliminating nuisance parameters |
| CLT | $\bar{X}_n \to \mathcal{N}(\mu, \sigma^2/n)$ | Justifying Gaussian approximations |

---

## 📝 Exercises

1. **Poisson counting:** A detector counts X-ray photons at a mean rate of λ = 3.7 photons/s. Compute P(X = 0), P(X ≤ 2), and P(X ≥ 5) over 1 second.

2. **Marginalisation:** Given joint PDF $f(x,y) = 6xy^2$ for $0 < x < 1$, $0 < y < 1$, find the marginal $f_X(x)$ and verify it integrates to 1.

3. **CLT check:** Generate 10,000 samples of the mean of n=30 Poisson(λ=2) variables. Plot the histogram and overlay the CLT Gaussian prediction. At what n does the approximation break down badly for λ=0.5?

4. **Conditional probability:** A galaxy survey contains 1000 galaxies: 300 spirals, 500 ellipticals, 200 irregulars. Of the spirals, 80% are star-forming; of ellipticals, 10%; of irregulars, 60%. If you randomly pick a star-forming galaxy, what is the probability it is a spiral?

---

## 🔗 Next Chapter

[→ Chapter 2: Bayes' Theorem and the Bayesian Framework](../ch02_bayes_theorem/README.md)