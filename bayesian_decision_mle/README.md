# Bayesian Decision Theory & Probabilistic Modeling

## Purpose

This project provides a formal exploration of fundamental concepts in Bayesian decision theory and maximum likelihood estimation (MLE). The objective is to compare decision strategies, derive optimal classification boundaries under Gaussian assumptions, and establish theoretical connections between probabilistic models and standard loss functions used in machine learning.

## Core Topics

- **Bayes versus randomized decision rules**
  Analysis of overall risk under zero‑one loss, demonstrating that the optimal Bayes classifier achieves lower or equal risk compared to any randomized rule. Conditions for equivalence are identified.

- **Gaussian classifiers**
  Derivation of decision boundaries for both univariate and multivariate Gaussian class‑conditional densities, examining the influence of prior probabilities and covariance structures.

- **Maximum likelihood estimation**
  MLE applied to Bernoulli (Naive Bayes) and Poisson distributions, showing that the resulting estimators correspond exactly to empirical frequencies observed in the data.

- **Linear regression and least squares**
  Formal proof that maximizing the log‑likelihood under the assumption of independent Gaussian errors is equivalent to minimizing the mean squared error (MSE) loss.

- **Naive Bayes classification**
  Practical implementation of a Naive Bayes classifier for binary features, including posterior score computation and label prediction.

## Implementation

The repository includes mathematical derivations (PDF) and supporting Python code (NumPy) for numerical verification of the theoretical results, such as computing conditional probabilities, estimating Poisson parameters, and running a Naive Bayes classifier on small binary datasets.
