# Lie-Geometric Trainability of Quantum Dynamical Systems
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

> Official implementation for the paper **Lie-Geometric Trainability of Quantum Dynamical Systems: Avoiding Barren Plateaus via Low-Dimensional Lie Subalgebras**
>
> *Physics Scripta (Accepted, 2025)*

---

## Overview
This repository contains all numerical experiments and code to verify the theoretical framework proposed in the paper.

The core conclusion:
The trainability of quantum dynamical systems is fundamentally determined by the **Lie closure** of Hamiltonian generator sets. Restricting the dynamics to a **polynomial-dimensional Lie subalgebra** can suppress exponentially vanishing gradients and effectively mitigate barren plateaus in quantum machine learning.

---

## Framework
<p align="center">
  <img src="framework.png" width="1000">
</p>

The core logical chain of this work:
**Generator Algebra**
↓
**Lie Closure Dimension**
↓
**Reachable Orbit Dimension**
↓
**Moment Scaling**
↓
**Gradient Variance**
↓
**Model Trainability**

---

## Main Theoretical Results
### 1. Lie-Restricted Model
For a quantum system constrained to a polynomial-dimensional Lie algebra $\mathfrak{g}_n$:
$$
\dim(\mathfrak g_n)=\mathrm{poly}(n)
$$
The gradient variance satisfies:
$$
\operatorname{Var}_{\theta}\left( \frac{\partial \mathcal L}{\partial \theta_k} \right) \ge \frac{C}{\mathrm{poly}(n)}
$$
This bound rules out barren plateaus and guarantees model trainability.

### 2. Full Expressive Model
For the full $\mathfrak{su}(2^n)$ algebra (fully expressive quantum model):
$$
\operatorname{Var}_{\theta}\left( \frac{\partial \mathcal L}{\partial \theta_k} \right) = O(2^{-n})
$$
The gradient decays exponentially with qubit number $n$, leading to severe barren plateaus.

### 3. Variance Enhancement Ratio
The variance ratio between Lie-restricted models and full models is:
$$
R_n = \frac{\operatorname{Var}_{\mathfrak g_n}(\partial_{\theta_k}\mathcal L)}{\operatorname{Var}_{\mathfrak{su}(2^n)}(\partial_{\theta_k}\mathcal L)} = \Omega\left( \frac{2^n}{\dim(\mathfrak g_n)} \right)
$$

---

## Repository Structure
