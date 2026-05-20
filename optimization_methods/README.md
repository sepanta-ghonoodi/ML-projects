# Optimization Methods

## Purpose

This project examines core optimization techniques that underpin modern machine learning, ranging from classical line search and gradient descent to Newton’s method, constrained optimization, and adaptive step‑size strategies. The focus is on understanding the theoretical properties and practical trade‑offs of each approach.

## Core Concepts

- **Line search** – Descent direction condition (negative gradient) and step‑length selection via exact line search, Armijo backtracking, Wolfe conditions, and fixed steps.

- **Bilevel optimization** – Leader‑follower problems where the lower level is solved analytically, then substituted into the upper objective; regularity conditions (convexity, uniqueness of the follower’s response).

- **Batch vs. stochastic gradient descent** – Trade‑offs between computational load per iteration, gradient noise (variance), and convergence stability versus the ability to escape local minima.

- **Steepest descent direction** – Proof using first‑order Taylor expansion that the negative gradient minimizes the linearized cost under a unit‑norm constraint.

- **Pseudoinverse and least squares** – Derivation showing that $x^* = (A^\top A)^{-1}A^\top b$ uniquely minimizes the squared residual norm for overdetermined systems with full column rank.

- **Quadratic convergence of Newton’s method** – Definition of quadratic convergence and demonstration that the Newton step approximates the exact displacement to a local minimizer when sufficiently close.

- **Equality‑constrained optimization** – Lagrangian formulation, analytic solution via stationarity conditions, and numerical solution using Newton‑Raphson on the resulting nonlinear system.

- **Adaptive step‑size strategies** – Armijo backtracking (sufficient decrease) versus AdaGrad (history‑based scaling), applied to the Rosenbrock function and compared via trajectories and convergence rates.

## Implementation

The repository includes mathematical derivations (PDF) and Python implementations of:

- Gradient descent with Armijo backtracking and AdaGrad
- Newton’s method for unconstrained and equality‑constrained problems
- Newton‑Raphson for solving the KKT system