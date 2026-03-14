# Chapter 2: Bayes' Theorem and the Bayesian Framework

> *"When the facts change, I change my mind. What do you do, sir?"* — John Maynard Keynes

---

## 2.1 The Frequentist vs. Bayesian Divide

Before deriving Bayes' theorem, we must understand *two fundamentally different philosophies* of probability:

### Frequentist Interpretation
Probability is the **long-run frequency** of an event in an infinite sequence of identical experiments.
$$P(A) = \lim_{n \to \infty} \frac{\text{number of times A occurs}}{n}$$

Consequences:
- Parameters are *fixed, unknown constants* — they have no probability distribution
- Data are random; parameters are not
- Confidence intervals: "95% of such intervals contain the true value" (not "95% probability the true value is in this interval")

### Bayesian Interpretation
Probability is a **degree of belief** in a proposition, given available information.

Consequences:
- Parameters *have* probability distributions, reflecting our uncertainty
- Beliefs are updated rationally as new data arrive
- Credible intervals: "There is 95% probability the true value lies in this interval" — exactly what scientists want to say

**In astrophysics:** We observe a gravitational wave signal once. We cannot repeat the merger 10,000 times. The Bayesian framework is the natural language for such single-event inference.

---

## 2.2 Derivation of Bayes' Theorem

Start from the definition of conditional probability:
$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$
$$P(B \mid A) = \frac{P(A \cap B)}{P(A)}$$

Both expressions contain $P(A \cap B)$. Equating:
$$P(A \mid B) \cdot P(B) = P(B \mid A) \cdot P(A)$$

Rearranging:

$$\boxed{P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}}$$

This is **Bayes' theorem**. Elementary algebra. Yet it is the foundation of a complete epistemology.

---

## 2.3 Bayes' Theorem for Statistical Inference

Replace A → θ (parameters) and B → D (data):

$$\boxed{P(\theta \mid D) = \frac{P(D \mid \theta) \cdot P(\theta)}{P(D)}}$$

Each term has a precise name and meaning:

| Term | Symbol | Name | Meaning |
|------|--------|------|---------|
| $P(\theta \mid D)$ | **posterior** | What we want | Updated belief about θ after seeing D |
| $P(D \mid \theta)$ | **likelihood** | How well does θ explain D? | Probability of observing D if θ were true |
| $P(\theta)$ | **prior** | What we believed before | Encodes prior knowledge / ignorance |
| $P(D)$ | **evidence** | Normalisation | Probability of data under all models |

---

## 2.4 The Evidence (Marginal Likelihood)

The denominator:
$$P(D) = \int P(D \mid \theta) \cdot P(\theta)\, d\theta$$

This is the **marginal likelihood** or **Bayesian evidence**. It:
1. Normalises the posterior so it integrates to 1
2. Is independent of θ (it's a number, not a function of θ)
3. Is critical for **model comparison** (Chapter 6)

For single-model inference, we often write:

$$\boxed{P(\theta \mid D) \propto P(D \mid \theta) \cdot P(\theta)}$$

*posterior ∝ likelihood × prior*

---

## 2.5 A Concrete Astrophysical Example: Is a Source Variable?

**Scenario:** You observe a star at two epochs. Flux measurements:
- Epoch 1: $F_1 = 100 \pm 5$ mJy
- Epoch 2: $F_2 = 115 \pm 5$ mJy

Is the star variable (θ = 1) or constant (θ = 0)?

**Prior:** 5% of stars of this type are variable.
$$P(\theta = 1) = 0.05, \quad P(\theta = 0) = 0.95$$

**Likelihood** (assuming Gaussian noise, independent measurements):

If constant: $\Delta F = F_2 - F_1 = 15$, combined uncertainty $\sigma = \sqrt{50} \approx 7.07$ mJy.

$$P(D \mid \theta = 1) \propto \exp\!\left(-\frac{15^2}{2 \cdot 50}\right) \approx 0.011 \quad \text{(some variability expected)}$$

Formally: evaluate a variability model vs. constant model — this is worked in full in Chapter 6.

---

## 2.6 Sequential (Online) Updating

A beautiful property of Bayes: **yesterday's posterior is today's prior**.

Start with prior $P_0(\theta)$. After observing datum $d_1$:
$$P_1(\theta) = P(\theta \mid d_1) \propto P(d_1 \mid \theta) \cdot P_0(\theta)$$

After observing $d_2$ (independent of $d_1$):
$$P_2(\theta) = P(\theta \mid d_1, d_2) \propto P(d_2 \mid \theta) \cdot P_1(\theta)$$

This is equivalent to updating on both data at once:
$$P(\theta \mid d_1, d_2) \propto P(d_1 \mid \theta) \cdot P(d_2 \mid \theta) \cdot P_0(\theta)$$

**Application:** Real-time updating of pulsar period as new pulse arrival times come in.

---

## 2.7 Point Estimates from the Posterior

Once we have the posterior $P(\theta \mid D)$, we can extract point estimates:

**Maximum A Posteriori (MAP):**
$$\hat{\theta}_{\text{MAP}} = \arg\max_\theta P(\theta \mid D) = \arg\max_\theta \left[\ln P(D \mid \theta) + \ln P(\theta)\right]$$

When the prior is uniform (flat): MAP = Maximum Likelihood Estimate (MLE).

**Posterior Mean:**
$$E[\theta \mid D] = \int \theta \cdot P(\theta \mid D)\, d\theta$$

**Posterior Median:** θ* such that $\int_{-\infty}^{\theta^*} P(\theta \mid D)\, d\theta = 0.5$

**Which to use?**
- MAP is easiest to compute but ignores asymmetry
- Mean minimises expected squared error
- Median is robust to outlier regions and skewed posteriors
- In astrophysics: **always report the full posterior when possible**

---

## 2.8 Credible Intervals

A **Bayesian credible interval** $[a, b]$ at level $1-\alpha$ satisfies:
$$\int_a^b P(\theta \mid D)\, d\theta = 1 - \alpha$$

**Highest Posterior Density (HPD) interval:** The shortest interval containing $1-\alpha$ probability mass. For unimodal, symmetric distributions this coincides with equal-tails.

$$\text{HPD}: \quad \{\theta : P(\theta \mid D) \geq k_\alpha\}$$

where $k_\alpha$ is chosen so the region contains $1-\alpha$ probability.

**Example:** Posterior on a black hole mass:
$$M_{\text{BH}} \in [6.8, 9.2]\, M_\odot \quad \text{(90% HPD)}$$

This means: *"Given our model and data, there is 90% probability the true mass lies in [6.8, 9.2] solar masses."*

---

## 2.9 Conjugate Priors

A prior $P(\theta)$ is **conjugate** to a likelihood $P(D \mid \theta)$ if the posterior has the same functional form as the prior.

This yields **analytic posteriors** — no integration required!

| Likelihood | Conjugate Prior | Posterior |
|-----------|----------------|-----------|
| Gaussian (known σ) | Gaussian(μ₀, σ₀²) | Gaussian |
| Poisson | Gamma(α, β) | Gamma |
| Binomial | Beta(α, β) | Beta |
| Multinomial | Dirichlet | Dirichlet |

### Example: Beta-Binomial (Detection Probability)

**Problem:** Estimate the efficiency ε ∈ [0,1] of a telescope pipeline.

**Data:** k = 73 injected signals recovered out of n = 100.

**Prior:** $\epsilon \sim \text{Beta}(\alpha_0, \beta_0)$, with:
$$P(\epsilon) = \frac{\epsilon^{\alpha_0-1}(1-\epsilon)^{\beta_0-1}}{B(\alpha_0, \beta_0)}$$

Uninformative choice: $\alpha_0 = \beta_0 = 1$ (uniform on [0,1]).

**Likelihood:** Binomial
$$P(k \mid \epsilon) = \binom{n}{k} \epsilon^k (1-\epsilon)^{n-k}$$

**Posterior:**
$$P(\epsilon \mid k) \propto \epsilon^{k + \alpha_0 - 1}(1-\epsilon)^{n-k+\beta_0-1} = \text{Beta}(\alpha_0 + k,\; \beta_0 + n - k)$$

With uniform prior: $\text{Beta}(74, 28)$

$$E[\epsilon \mid k] = \frac{74}{74+28} \approx 0.726, \qquad \text{Mode} = \frac{73}{100} = 0.73$$

95% credible interval: $[0.634, 0.806]$

---

## 2.10 The Likelihood Principle

**Theorem (Birnbaum, 1962):** If two experiments yield proportional likelihood functions for θ, all evidence about θ is identical, regardless of the experiment design.

$$\mathcal{L}_1(\theta; D_1) \propto \mathcal{L}_2(\theta; D_2) \implies \text{same inference}$$

Bayesian inference automatically satisfies this principle. Frequentist procedures (p-values) do not.

**Consequence:** Stopping rules don't matter in Bayesian analysis. If you had planned to observe 100 stars but stopped at 73 due to weather, the inference is the same as if you had planned to stop at 73.

---

## 2.11 Summary

```
Prior Belief  +  New Data  →  Updated Belief
   P(θ)       ×  P(D|θ)   ∝     P(θ|D)
```

The Bayesian framework:
1. **Encodes** prior knowledge (physical constraints, previous experiments)
2. **Updates** beliefs coherently when new data arrive
3. **Propagates** uncertainty through all derived quantities
4. **Provides** probabilistic statements that match scientific intuition

---

## 📝 Exercises

1. **Beta-Binomial:** You test a pulsar detection algorithm on 50 injections, recovering 42. With a uniform prior, compute the posterior distribution on detection efficiency. Find the 68% and 95% HPD intervals.

2. **Sequential updating:** Observe 3 flux measurements from a quasar: 10.2, 10.8, 9.9 mJy. Model the true flux as Gaussian with known σ = 0.5 mJy. Start with prior N(10, 1²). Update sequentially after each measurement. Show the prior, posterior-after-1, and posterior-after-3 on the same plot.

3. **MAP vs Mean:** Generate 1000 samples from a skewed Beta(2, 8) posterior. Compute the MAP, mean, and median. Which best summarises the distribution for reporting?

4. **Frequentist vs Bayesian:** A significance test yields p = 0.04 for a claimed X-ray transient. Why does this not mean "4% probability the source is noise"? What Bayesian quantity does answer this question?

---

## 🔗 Navigation

[← Chapter 1: Probability Foundations](../ch01_probability_foundations/README.md)  
[→ Chapter 3: Likelihood Functions and Prior Selection](../ch03_likelihood_priors/README.md)