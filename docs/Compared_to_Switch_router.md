# Numerical Comparison: Oblivious Bit-Packed Phase Router vs Switch Transformer Routing

This document compares the **Oblivious Bit-Packed Phase Router** with **Switch Transformer–style routing** along two quantitative axes:

- **Load variance**
- **Collisions and drops under capacity constraints**

The focus is strictly on _routing quality and concentration properties_, not semantic expressiveness.

---

## 1. Problem Setup

Let:

- **N** — number of tokens
- **E** — number of experts
- **k** — routes per token
- Total assignments: \( N \cdot k \)

The ideal mean load per expert is:

\[
\mu = \frac{N k}{E}
\]

Assumptions:

- Dense or moderately dense routing regime
- No pathological sparsity
- Switch routing uses **top-k softmax**
- The bit-packed router uses **randomized oblivious matching**
- Capacity limits are introduced explicitly in Section 3

---

## 2. Load Variance

### 2.1 Switch Transformer Routing

Switch routing assigns experts via learned logits followed by top-k selection.

Key characteristics:

- Routing decisions are **correlated across tokens**
- Similar token representations yield similar expert choices
- Load balance is enforced only via auxiliary losses

Empirical and theoretical behavior:

\[
\mathrm{Var}\_{\text{Switch}} \approx c_1 \cdot \mu^2,
\qquad c_1 \in [0.2, 1.0]
\]

Implications:

- Variance grows **quadratically** with mean load
- Expert loads exhibit heavy tails
- Overloads are common without explicit balancing mechanisms

This behavior is well documented in the Switch and GShard literature.

---

### 2.2 Oblivious Bit-Packed Phase Router

The oblivious router performs:

1. Deterministic row rotations
2. Random column permutations
3. Bitwise intersection
4. Uniform random selection of up to \( k \) matches

This approximates **independent Bernoulli trials** per expert:

\[
L_j \sim \text{Binomial}\!\left(N, \frac{k}{E}\right)
\]

Hence:

\[
\mathbb{E}[L_j] = \mu
\qquad
\mathrm{Var}\_{\text{Oblivious}} = \mu \left(1 - \frac{k}{E}\right) \approx \mu
\]

Implications:

- Variance grows **linearly** with mean load
- Loads concentrate sharply around the mean
- No learned collapse or correlation effects

---

### 2.3 Numerical Example

Assume:

- \( N = 65{,}536 \)
- \( E = 256 \)
- \( k = 2 \)

Then \( \mu = 512 \).

| Router               | Std. Deviation |
| -------------------- | -------------- |
| Switch (typical)     | 200–500        |
| Oblivious bit-packed | ~22            |

The oblivious router achieves **order-of-magnitude tighter concentration**.

---

## 3. Collisions and Drops (With Capacity Limits)

Assume each expert has capacity \( C \).

---

### 3.1 Switch Routing

Switch routing assigns experts first and enforces capacity afterward:

- Overloaded experts drop tokens
- Drops are mitigated via auxiliary losses and over-provisioning

Observed behavior:

- No aux loss: **5–30% drop rate**
- With aux loss: **1–5% drop rate**
- Drops persist, especially early in training

Since:
\[
\sigma^2 = O(\mu^2),
\]
the tail probability of overload remains large.

---

### 3.2 Oblivious Bit-Packed Phase Router

The oblivious router:

- Explicitly enumerates matches
- Never silently over-assigns
- Emits `-1` when no valid route exists

Using Chernoff bounds for binomial load:

\[
\Pr[L_j > (1+\delta)\mu]
\le
\exp\!\left(-\frac{\delta^2 \mu}{3}\right)
\]

Example:

- \( \mu = 500 \)
- \( \delta = 0.2 \)

\[
\Pr \lesssim e^{-6.6} \approx 0.0013
\]

Across all experts:
\[
\Pr(\text{any overflow}) \approx E \cdot 0.0013
\]

Overflow probability decays **exponentially** with mean load.

---

## 4. Correlation Effects

### Switch Routing

- Logits are correlated across tokens
- Tokens from the same sequence cluster on experts
- Burst overloads are common

### Oblivious Bit-Packed Phase Router

- Random permutations and rotations destroy correlation
- Identical rows still route independently
- Load remains stable across batches

Correlation suppression is a primary source of improved concentration.

---

## 5. Summary Table

| Property          | Switch Routing | Oblivious Bit-Packed Router |
| ----------------- | -------------- | --------------------------- |
| Routing type      | Learned        | Oblivious                   |
| Variance scaling  | \( O(\mu^2) \) | \( O(\mu) \)                |
| Load tails        | Heavy          | Exponentially suppressed    |
| Drop rate         | 1–30%          | Near zero                   |
| Capacity handling | Reactive       | Intrinsic                   |
| Determinism       | No             | Yes                         |
| Training required | Yes            | No                          |

---

## 6. Bottom Line

Numerically and probabilistically:

- Switch routing exhibits quadratic variance and frequent overloads.
- The Oblivious Bit-Packed Phase Router achieves linear variance and exponentially small overflow probability.
- These guarantees hold **without training, auxiliary losses, or token dropping**.

Switch routing excels in **semantic adaptivity**.

The Oblivious Bit-Packed Phase Router excels in **balance, predictability, and hardware efficiency**.

> _This router trades semantic optimality for provable concentration and deterministic performance—and wins decisively on load balance._

---

## 7. Are Both Routers Oblivious?

**Short answer:** no.  
Only the Oblivious Bit-Packed Phase Router is oblivious in the formal routing-theoretic sense.

---

### 7.1 Formal Definition

In routing theory (Valiant, Leighton, Räcke), a routing algorithm is **oblivious** if routing decisions are independent of:

- Current congestion
- Traffic patterns
- Other packets’ routing decisions

Routes depend only on fixed randomness and static structure, yielding guarantees that hold for _all_ traffic matrices.

---

### 7.2 Oblivious Bit-Packed Phase Router

Routing decisions depend only on:

- Static bit patterns
- Deterministic rotations
- Fixed permutations
- Per-row seeded randomness

Formally:
\[
\text{Route}(i) = f(S_i, T, \pi, \text{seed})
\]

There is no dependence on load, contention, or feedback.

**Conclusion:** the router is genuinely oblivious.

---

### 7.3 Switch Transformer Routing

Despite being stateless at inference time, Switch routing is **not oblivious**.

Violations include:

1. Learned routing weights encoding historical traffic
2. Correlated routing decisions across tokens
3. Load-aware auxiliary losses
4. Reactive capacity enforcement

Formally:
\[
\text{Route}(i) = f(h_i, W(\text{past traffic}), \text{current load})
\]

This is adaptive routing.

---

### 7.4 Classification Summary

| Router                      | Oblivious? | Reason                 |
| --------------------------- | ---------- | ---------------------- |
| Oblivious bit-packed router | ✅ Yes     | Traffic-independent    |
| Valiant routing             | ✅ Yes     | Randomized             |
| Beneš network               | ✅ Yes     | Fixed permutation      |
| Switch Transformer          | ❌ No      | Learned, load-reactive |
| TCAndon-style routers       | ❌ No      | Semantic, adaptive     |

---

## 8. When Is Non-Oblivious Routing Better?

Oblivious routing offers strong worst-case guarantees, but adaptivity can dominate in certain regimes.

### Non-oblivious routing is preferable when:

- Experts are **semantically specialized**
- Traffic is **structured or repetitive**
- Expert counts are small and capacity is tight
- Training dynamics and co-adaptation matter
- Latency-critical, small-batch inference dominates

Adaptive routing exploits structure and improves average-case performance, at the cost of worst-case guarantees.

---

### Practical Takeaway

- Use **oblivious routing** for predictability, balance, and hardware efficiency.
- Use **non-oblivious routing** for semantic specialization and training-time gains.
- Hybrid designs—adaptive proposal with oblivious enforcement—often provide the best trade-off.
