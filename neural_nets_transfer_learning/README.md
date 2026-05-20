# Optimization, Neural Networks & Transfer Learning

## Purpose

This project explores key optimization methods (MLE, SGD, Newton’s method), model complexity (bias‑variance trade‑off), perceptron limitations, neural computation graphs, loss functions for noisy labels, the dead ReLU problem, Xavier initialization, and transfer learning on CIFAR‑10 using pre‑trained VGG‑11.

## Core Concepts

- **Optimization & regularization** – MLE for power‑law distributions; SGD update for logistic regression; Newton’s method converges in one step for L2‑regularized linear regression. Polynomial regression shows underfitting vs. overfitting; ridge regularization controls effective complexity.

- **Perceptron & feature transformations** – Linear classifier fails on concentric (non‑separable) data. Adding squared terms or Manhattan distance enables separation.

- **Neural networks & loss functions** – Computation graphs (linear, affine, ReLU, sums of ReLUs) represent piecewise linear functions. Cross‑entropy equals negative log‑likelihood; robust loss handles label noise; for targets ±1 use tanh output.

- **Weight initialization & dead ReLU** – Xavier initialization balances forward/backward variance; small gain errors cause exponential variance growth with depth (risks FP16 overflow). Dead ReLU: neurons with very negative bias rarely activate, yielding zero gradient permanently.

- **Transfer learning** – Pre‑trained VGG‑11 (frozen or fine‑tuned) significantly outperforms a baseline MLP on CIFAR‑10 due to spatial inductive bias (locality, translation invariance).

## Implementation

- Python implementations of MLE derivation, SGD for logistic regression, and polynomial regression with bias‑variance plots
- Perceptron with custom feature transformations, ReLU computation graphs, and dead ReLU probability simulation
- Transfer learning experiments on CIFAR‑10: MLP baseline vs. frozen VGG‑11 vs. fine‑tuned VGG‑11
