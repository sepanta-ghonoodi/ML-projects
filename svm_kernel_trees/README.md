# SVM, Kernel Methods & Decision Trees

## Purpose

This project explores Support Vector Machines (kernel trick, dual formulation, soft‑margin), decision tree induction (ID3 algorithm), and their practical implementation on real datasets.

## Core Concepts

- **SVM dual vs. primal** – Dual formulation preferred for kernels: optimization depends on number of samples (not feature dimension), and kernel trick replaces dot products via Mercer’s theorem, enabling infinite‑dimensional feature spaces without explicit mapping.

- **RBF kernel** – Maps data into an infinite‑dimensional Hilbert space (proved via Taylor expansion of the exponential).

- **Explicit cubic feature map** – Expansion of $(1 + x^\top z)^3$ yields a 10‑dimensional feature space with combinatorial scaling factors.

- **ID3 decision trees** – Greedy top‑down construction using information gain (always non‑negative due to concavity of entropy). Handles discrete and continuous features (optimal split threshold).

- **Greedy trap (XOR)** – XOR gives zero gain at root; ID3 fails to start despite existence of a perfect depth‑2 tree. Illustrates preference bias over restriction bias.

- **Class imbalance** – In medical datasets (Pima Indians), standard SVM biases toward majority class; balanced SVM improves recall from ~0.49 to ~0.70.

- **Primal soft‑margin SVM** – Solved as a quadratic programming problem with slack variables $\xi_i$ to allow misclassifications.

- **Overfitting & pruning** – Noise injection drastically increases tree size; pruning trades off training vs. test accuracy to improve generalization.

## Implementation

Python implementations of: SVM pipeline (scaling, kernels, grid search, PCA, class balancing), primal SVM via QP (cvxpy), and complete ID3 from scratch (entropy, information gain, continuous splitting, pruning simulation).