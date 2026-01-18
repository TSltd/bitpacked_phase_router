# **Orthogonal Load-Balanced Incidence Operator (OLBIO)**

_A Deterministic, Phase-Based, Structured Mixing Operator for Balanced Incidence Structures_

---

## **1. Formal Definitions**

### **1.1 Input Domain**

Let:

```
S, T ∈ {0,1}^{N × N}
```

- `S[i,j] = 1` indicates a potential connection from source `i` to bucket `j`.
- `T[i,j] = 1` indicates a potential connection from bucket `j` to sink `i`.

Define row and column sums:

```
s_i = sum_j S[i,j]      # row degree of source i
t_j = sum_i T[i,j]      # column degree of target j
```

The goal of OLBIO is to produce a sparse incidence matrix `O ∈ {0,1}^{N × N}` such that:

1. **Fan-out constraint**:

```
forall i: sum_j O[i,j] ≤ k
```

2. **Load balance**:

```
max_j L_j ≈ mean_j L_j,  where  L_j = sum_i O[i,j]
```

3. **Degree-preserving in expectation**:

```
E[L_j] ∝ t_j
```

---

## **2. High-Level Pseudocode**

```
function OLBIO(S, T, k, seed):
    # Step 0: Left-align rows
    S_aligned = left_align(S)
    T_aligned = left_align(T)

    # Step 1: Phase spreading
    phi_S = cumulative_offsets(S_aligned)
    phi_T = cumulative_offsets(T_aligned)

    S_rot = apply_phase_spread(S_aligned, phi_S)
    T_rot = apply_phase_spread(T_aligned, phi_T)

    # Step 2: Independent column permutations
    col_perm_S = permute_indices(seed ^ 0xA5A5A5)
    col_perm_T = permute_indices(seed ^ 0xC6BC27)

    S_shuf = permute_columns(S_rot, col_perm_S)
    T_shuf = permute_columns(T_rot, col_perm_T)

    # Step 3: Optional row permutations
    row_perm_S = permute_indices(seed ^ 0x9E3779B)
    row_perm_T = permute_indices(seed ^ 0xD192ED03)

    S_final = permute_rows(S_shuf, row_perm_S)
    T_prepared = permute_rows(T_shuf, row_perm_T)

    # Step 4: Transpose T
    T_final = rotate90_clockwise(T_prepared)

    # Step 5: Intersection and top-k selection
    O = zeros(N,N)
    for i in 0…N-1:
        candidates = indices_of(S_final[i] AND T_final[i])
        chosen = deterministic_top_k(candidates, k, seed ^ i)
        O[i, chosen] = 1

    return O
```

---

## **3. Key Properties**

### **3.1 Bounded Fan-out**

```
sum_j O[i,j] ≤ k
```

Holds deterministically.

---

### **3.2 Degree Preservation in Expectation**

For a fixed column `j`:

```
Pr(S'_ik = 1) ≈ s_i / N
Pr(T'_jk = 1) ≈ t_j / N
```

Then:

```
E[O_ij] ≈ sum_{k=1}^{N} (s_i * t_j / N^2) = s_i * t_j / N
```

Column load expectation:

```
E[L_j] = sum_i E[O_ij] = (t_j / N) * sum_i s_i
```

Matches the Chung–Lu expected degree model.

---

### **3.3 Low Skew / Load Balancing**

- Phase spreading distributes 1s evenly.
- Column and row permutations break geometric correlations.
- Intersection produces weakly dependent Bernoulli sums → low variance.
- Empirical result:

```
max_j L_j ≈ mean_j L_j
```

---

### **3.4 Deterministic Reproducibility**

All randomness is seed-deterministic:

```
OLBIO(S, T, k, seed) == OLBIO(S, T, k, seed)
```

---

## **4. Applications**

- **Balanced Bipartite Sampling:** Large bipartite graphs with prescribed degrees.
- **Deterministic Sparse Coding:** Compressed sensing, feature hashing, binary embeddings.
- **Load-Aware Scheduling:** Task assignment with bounded peaks and predictable distribution.
- **Multi-Choice Hash Mapping:** Deterministic k-candidate bucket selection, reducing collisions.
- **Multi-Constraint Assignment:** Extending to 3+ dimensions for balanced assignment across axes.

---

## **5. Limitations**

| Limitation              | Explanation                                                         |
| ----------------------- | ------------------------------------------------------------------- |
| Slower than single hash | Phase-based rotations and permutations incur overhead               |
| Global view required    | Cannot map keys independently without full matrices                 |
| Memory traffic          | Bit-packed operations require more memory access than a simple hash |

**Best fit:** Use cases where **load balance and reproducibility** are more critical than raw speed.

---

## **6. Intuition Recap**

> OLBIO transforms sparse incidence patterns via orthogonal, phase-based mixing to produce balanced, deterministic, bounded incidence structures.

---

## **7. Pseudo-Proof Sketch (Load Concentration)**

Let `X_ij^(k)` be indicator variables:

```
X_ij^(k) = S'_ik AND T'_jk
```

- Under phase spreading and permutation: weakly dependent Bernoulli with

```
Pr(X_ij^(k) = 1) ≈ s_i * t_j / N^2
```

- Sum over k:

```
E[O_ij] = sum_{k=1}^{N} Pr(X_ij^(k) = 1) ≈ s_i * t_j / N
```

- Column load concentration:

```
Pr(|L_j - E[L_j]| > ε) ≤ 2 exp(-c ε^2 N)
```

---

## **8. Example: Deterministic Top-k Selection**

```cpp
std::mt19937_64 rng(seed ^ i);
std::shuffle(candidates.begin(), candidates.end(), rng);
for (size_t cnt = 0; cnt < k && cnt < candidates.size(); cnt++)
    routes[i*k + cnt] = candidates[cnt];
```

---

## **9. Implementation Notes**

- Bit-packed storage enables SIMD operations.
- Rotations: word-level shifts for O(N) per row.
- Permutations are computed once per seed.

---

## **10. Summary**

OLBIO provides:

- **Bounded fan-out**
- **Low skew**
- **Deterministic reproducibility**
- **Applicability to hashing, routing, scheduling, and sparse coding**

It sits between **random hashing** and **algorithmic combinatorial design**.

---
