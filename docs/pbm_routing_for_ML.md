# PBM Router for Machine Learning

This script provides a method to extract **structural features from binary PBM images** using a specialized routing algorithm. These features can be used for **contrastive learning, clustering, classification, or similarity-based retrieval**.

## Overview

The pipeline converts each PBM into a **row-aligned routed feature vector**, capturing structural patterns beyond simple pixel-wise comparison. The routing process encodes **row-wise bit overlaps**, producing deterministic and comparable vectors for all images.

Key steps:

1. **Load PBMs:** Convert each image into a 2D binary array of 0s and 1s.
2. **Left-align rows:** All “ones” are shifted to the left, creating consistent row-wise structure.
3. **Self-route images:** Each PBM is routed with itself to produce a **structural feature vector**.
4. **Flatten & normalize:** Routed arrays are converted to 1D normalized vectors for ML use.
5. **Compute similarities:** Cosine similarity between vectors captures **structural similarity** between images.
6. **Reuse routed arrays:** Each PBM is routed only once for efficiency.

---

## Advantages

- **Structural similarity features:** Captures row-aligned patterns, not just raw pixels.
- **Shift and noise tolerance:** Left-alignment makes features robust to small translations and sparse noise.
- **Fixed-size vectors:** Each PBM produces a deterministic, flattened vector suitable for ML models.
- **Efficient:** Self-routing each PBM once allows scaling to large datasets.
- **Similarity matrix:** Enables graph-based learning, contrastive loss training, clustering, or nearest-neighbor retrieval.

---

## Example ML Workflows

### 1. Contrastive Learning

- Positive pairs: same PBM under minor transformations.
- Negative pairs: different PBMs.
- Train a network to maximize similarity for positives and minimize similarity for negatives.
- Result: embeddings that capture **structural patterns** robustly.

### 2. Clustering

- Compute pairwise similarity between all PBMs.
- Apply clustering algorithms (hierarchical, spectral, etc.) to group **structurally similar images**.

### 3. Feature-based Classification

- Use routed vectors as **input features** for classifiers (SVM, logistic regression, MLP).
- Learn to predict classes based on **row-aligned bit patterns**.

---

## Usage

```bash
python pbm_router_pairwise_optimized.py ./pbm_images/
```

- Prints similarity for each PBM pair.
- Produces a **full pairwise similarity matrix**.
- Routed arrays are computed once per PBM for efficiency.

---

## Notes

- Cosine similarity ensures **scale-invariant comparison**.
- This method is especially suitable for **binary or PBM-style images**.
- Precomputed routed arrays can be **cached and reused** to avoid repeated routing in large datasets.

---
