# Unsupervised Learning, Dimensionality Reduction, and Feature Selection

## Purpose

This project explores unsupervised learning (clustering), dimensionality reduction (PCA, Kernel PCA, LDA), and feature selection methods (filter, wrapper, embedded). It includes manual computations, theoretical proofs, and practical implementations on synthetic and real datasets.

## Core Concepts

- **K‑Means clustering** – One full iteration with Euclidean distance: assignment step, centroid update, and cardinality reporting.

- **Agglomerative clustering linkages** – Comparison of single vs. complete linkage in terms of computational complexity ($O(N^2)$ to $O(N^3)$) and robustness to outliers (complete linkage is more robust). Complete linkage dendrogram construction on a 9‑point dataset.

- **Dimensionality reduction choice** – When to use PCA (linear Gaussian data, noise filtering), Kernel PCA (nonlinear structures like concentric circles), and LDA (supervised class separation with overlap).

- **LDA rank proof** – Between‑class scatter matrix $S_B$ has rank at most $C-1$, therefore at most $C-1$ discriminant directions.

- **Kernel PCA derivation** – Covariance in feature space is infinite‑dimensional; principal components are linear combinations of mapped samples; solving $K\alpha = N\lambda\alpha$; projection via kernel evaluations.

- **Manual LDA computation** – Two‑class synthetic data: class means, within/between scatter matrices, Fisher discriminant direction, and 1D projection.

- **Feature selection strategies** – Sequential Forward/Backward Selection (greedy, nesting effect, suboptimality example with XOR‑like interaction), bidirectional selection, comparison with RFE and Lasso (complexity, correlation handling, high‑dimensional suitability).

- **PCA on Iris** – Eigenvalues, explained variance (95% in first two PCs), class separability (Setosa perfect, others overlap). Comparison with LDA.

- **Kernel PCA on concentric circles** – Linear PCA fails; RBF kernel maps data to a space where circles become separable.

- **Feature selection on real dataset** – Correlation filtering, univariate selection vs. RFE, RFECV (optimal subset size 11), Lasso sparsity (age coefficient zero), PCA inefficiency (12 components for 95% variance).

- **Image segmentation (bonus)** – K‑Means vs. GMM on 5D features (spatial + RGB). K‑Means gives hard Voronoi boundaries; GMM yields softer, covariance‑aware segments with more organic shapes.

## Implementation

Python implementations include:

- Manual K‑Means iteration and complete linkage dendrogram
- PCA, Kernel PCA (RBF), and LDA on synthetic/real datasets
- Sequential feature selection, RFE, RFECV, and Lasso
- Image segmentation with K‑Means and GMM (spatial‑color features)
