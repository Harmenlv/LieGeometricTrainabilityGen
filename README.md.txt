# Lie-Geometric Trainability of Quantum Dynamical Systems

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

Official implementation for:

> **Lie-Geometric Trainability of Quantum Dynamical Systems:
Avoiding Barren Plateaus via Low-Dimensional Lie Subalgebras**

Physics Scripta (Accepted, 2025)

---

# Overview

This repository provides numerical experiments supporting the theoretical framework developed in the paper:

> The trainability of a quantum dynamical system is governed by the Lie closure of its generator set.

The central result is that restricting dynamics to a polynomial-dimensional Lie algebra prevents exponential gradient concentration and mitigates barren plateaus.

---

# Framework

<p align="center">
  <img src="framework.png" width="1000">
</p>

The framework establishes the following chain:

Generator Algebra

↓

Lie Closure Dimension

↓

Reachable Orbit Dimension

↓

Moment Scaling

↓

Gradient Variance

↓

Trainability

---

## Main Theoretical Result

For a Lie-restricted model

\[
\dim(\mathfrak g_n)=\mathrm{poly}(n),
\]

the gradient variance satisfies

\[
\operatorname{Var}_{\theta}
\left(
\frac{\partial \mathcal L}
{\partial \theta_k}
\right)
\ge
\frac{C}{\mathrm{poly}(n)},
\]

which excludes barren plateaus.

In contrast, fully expressive models satisfy

\[
\operatorname{Var}_{\theta}
\left(
\frac{\partial \mathcal L}
{\partial \theta_k}
\right)
=
O(2^{-n}),
\]

leading to exponential gradient concentration.

The variance enhancement factor scales as

\[
R_n
=
\frac{
\operatorname{Var}_{\mathfrak g_n}
(\partial_{\theta_k}\mathcal L)
}{
\operatorname{Var}_{\mathfrak{su}(2^n)}
(\partial_{\theta_k}\mathcal L)
}
=
\Omega
\left(
\frac{2^n}
{\dim(\mathfrak g_n)}
\right).
\]

---

# Repository Structure

```text
LieGeometricTrainability/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── framework.png
├── Generator_Structure.png
│
├── Main.py
├── Main_con.py
│
├── experiments/
│   │
│   ├── appendix_variance_experiment.py
│   ├── appendix_variance_experiment4.py
│   ├── appendix_variance_experiment5.py
│   │
│   ├── collective_spin_variance_experiment.py
│   └── initial_residual_scaling_experiment.py
│
├── data/
│   │
│   ├── lie_variance_results.csv
│   ├── full_variance_results.csv
│   ├── collective_spin_variance_results.csv
│   └── initial_residual_scaling_results.csv
│
├── figures/
│   │
│   ├── collective_spin_variance_summary.png
│   └── initial_residual_scaling_summary.png
│
└── docs/
    └── paper.pdf