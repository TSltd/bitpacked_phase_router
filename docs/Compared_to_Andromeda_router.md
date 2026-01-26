# Comparison: Oblivious Bit-Packed Phase Router vs Andromeda-Style Routers

This document compares the **Oblivious Bit-Packed Phase Router** with **Andromeda-style routing mechanisms** used in large-scale Mixture-of-Experts (MoE) systems.  
The comparison focuses on **routing behavior, concentration guarantees, adaptivity, and systems implications**, rather than model quality.

---

## 1. Background

### 1.1 Oblivious Bit-Packed Phase Router

The Oblivious Bit-Packed Phase Router is a **traffic-independent routing mechanism** that assigns tokens to experts using:

- Static bit patterns
- Deterministic row rotations
- Random column permutations
- Bitwise intersections followed by uniform selection

Routing decisions depend only on fixed randomness and static structure, not on expert load, token correlations, or learned parameters.

---

### 1.2 Andromeda-Style Routers

Andromeda-style routers (including TC-Andon and related designs) are **learned or semi-learned routing systems** characterized by:

- Token embeddings projected into a routing space
- Hashing, quantization, or codebook lookup
- Learned centroids, tables, or routing networks
- Explicit or implicit load balancing constraints

These routers aim to **align tokens with semantically specialized experts** while controlling load imbalance through adaptive mechanisms.

---

## 2. Routing Model Comparison

| Aspect                    | Bit-Packed Phase Router    | Andromeda-Style Router    |
| ------------------------- | -------------------------- | ------------------------- |
| Routing type              | Oblivious                  | Adaptive / learned        |
| Dependence on load        | None                       | Explicit or implicit      |
| Dependence on training    | None                       | Required                  |
| Randomization             | Fixed, traffic-independent | Learned or data-dependent |
| Correlation across tokens | Suppressed                 | Often high                |

---

## 3. Load Variance and Concentration

### 3.1 Bit-Packed Phase Router

Under standard assumptions, expert load behaves approximately as:

\[
L_j \sim \text{Binomial}\left(N, \frac{k}{E}\right)
\]

Yielding:

\[
\mathbb{E}[L_j] = \mu, \qquad \mathrm{Var}(L_j) \approx \mu
\]

Properties:

- Linear variance scaling
- Exponential tail suppression
- Strong batch-to-batch stability

These guarantees hold **independently of token distribution**.

---

### 3.2 Andromeda-Style Routers

Andromeda-style routers exhibit:

- Correlated routing decisions
- Learned clustering of tokens
- Load distributions dependent on data statistics

Empirically:

\[
\mathrm{Var}(L_j) = \Theta(\mu^2)
\quad \text{(without strong regularization)}
\]

Load balance is maintained via:

- Auxiliary losses
- Capacity factors
- Token dropping or rerouting

Concentration guarantees are **distribution-dependent**, not worst-case.

---

## 4. Collisions, Capacity, and Failure Modes

### 4.1 Bit-Packed Phase Router

- Capacity is enforced intrinsically
- Over-assignment is impossible by construction
- Routing failure is explicit (`-1`), not silent
- Overflow probability decays exponentially with mean load

This yields predictable performance and bounded tail latency.

---

### 4.2 Andromeda-Style Routers

- Capacity enforcement is reactive
- Experts may overload before mitigation
- Token dropping or reassignment is common
- Tail latency depends on training quality and data skew

Failure modes are often **soft but frequent**.

---

## 5. Correlation and Stability

### Bit-Packed Phase Router

- Random permutations decorrelate identical or similar tokens
- No feedback loops or reinforcement effects
- Stable load across batches and sequences

### Andromeda-Style Routers

- Similar embeddings cluster on experts
- Temporal and batch-level correlations accumulate
- Susceptible to burst overloads and collapse modes

---

## 6. Adaptivity and Semantic Alignment

### Bit-Packed Phase Router

- Experts are treated as interchangeable
- No semantic alignment between tokens and experts
- Optimal when experts implement homogeneous computation

---

### Andromeda-Style Routers

- Experts specialize via training
- Routing aligns tokens to expert semantics
- Enables conditional computation and parameter efficiency

This is the **primary advantage** of Andromeda-style routing.

---

## 7. Training Dynamics

| Aspect                     | Bit-Packed       | Andromeda-Style               |
| -------------------------- | ---------------- | ----------------------------- |
| Router training            | None             | Required                      |
| Expert specialization      | Fixed / external | Emergent                      |
| Collapse modes             | None             | Common without regularization |
| Hyperparameter sensitivity | Low              | High                          |

Andromeda-style routers rely on careful tuning to avoid imbalance and collapse.

---

## 8. Systems and Hardware Considerations

### Bit-Packed Phase Router

- Deterministic and reproducible
- Bitwise operations, SIMD-friendly
- Low control-flow divergence
- Predictable memory access patterns

---

### Andromeda-Style Routers

- Embedding projections and lookups
- Branching and dynamic behavior
- Load imbalance impacts utilization
- Higher variance in latency

---

## 9. Summary Table

| Dimension             | Bit-Packed Phase Router | Andromeda-Style Router |
| --------------------- | ----------------------- | ---------------------- |
| Oblivious routing     | Yes                     | No                     |
| Load variance         | \( O(\mu) \)            | \( O(\mu^2) \)         |
| Worst-case guarantees | Strong                  | Weak                   |
| Semantic adaptivity   | None                    | High                   |
| Training complexity   | None                    | High                   |
| Hardware efficiency   | High                    | Moderate               |
| Failure behavior      | Explicit                | Implicit               |

---

## 10. When Each Is Preferable

**Use the Bit-Packed Phase Router when:**

- Experts are homogeneous
- Predictable load and latency matter
- Large batches dominate
- Worst-case guarantees are required

**Use Andromeda-style routing when:**

- Experts are semantically specialized
- Training-time co-adaptation is desired
- Average-case performance dominates
- Conditional computation is the goal

---

## 11. Hybrid Outlook

In practice, the strongest systems often combine both approaches:

- **Adaptive routing to propose expert sets**
- **Oblivious routing to enforce balance and capacity**

This hybrid model preserves semantic gains while restoring strong concentration guarantees.

---

## Conclusion

The Oblivious Bit-Packed Phase Router and Andromeda-style routers represent **fundamentally different design points**:

- One optimizes for **worst-case concentration and determinism**
- The other optimizes for **semantic alignment and average-case efficiency**

Understanding this trade-off is essential when choosing a routing strategy for large-scale MoE systems.
