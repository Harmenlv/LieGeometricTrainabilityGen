#!/usr/bin/env python3
"""
Initial-residual scaling experiment for the collective-spin Lie-QNN.

Purpose
-------
This script provides numerical support for the revised proof step that assumes
an n-independent lower bound on the initial training residual.  It does not
replace the theoretical justification; it checks that the benchmark protocol
used in the revised manuscript does not accidentally make the initial residual
vanish as n grows.

For each system size n, the script fixes normalized collective observables

    O_z = 2 J_z / n,    O_x = 2 J_x / n,

generates a target from the same collective-spin Lie model, samples random
initial parameters theta, and estimates

    E_theta[(O_z(theta) - y_z)^2],
    E_theta[(O_x(theta) - y_x)^2],
    E_theta[||f(theta) - y||_2^2].

The default teacher is deterministic and n-independent: it applies a pi/2
collective rotation about J_y, giving a target close to (O_z, O_x) = (0, 1)
for every n.  The zero-parameter prediction is (1, 0), so the protocol has a
clear O(1) residual margin before random sampling is even considered.

PyCharm-friendly defaults
-------------------------
Running this file directly is equivalent to:

    python initial_residual_scaling_experiment.py \
        --n-values 3 4 5 6 8 10 12 16 20 24 32 \
        --theta-samples 1000 \
        --lie-layers 2 \
        --angle-mode uniform \
        --teacher-mode deterministic_margin
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm


Array = np.ndarray


@dataclass
class ResidualResult:
    n: int
    symmetric_dim: int
    lie_params: int
    theta_samples: int
    target_oz: float
    target_ox: float
    zero_oz: float
    zero_ox: float
    zero_residual_sq: float
    mean_vector_residual_sq: float
    std_vector_residual_sq: float
    median_vector_residual_sq: float
    q05_vector_residual_sq: float
    q95_vector_residual_sq: float
    min_vector_residual_sq: float
    max_vector_residual_sq: float
    mean_oz_residual_sq: float
    mean_ox_residual_sq: float
    witness_observable: str
    witness_beta_estimate: float
    elapsed_sec: float


def format_eta(seconds: float) -> str:
    if not np.isfinite(seconds):
        return "?"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m"
    return f"{minutes:d}m {sec:02d}s"


def build_collective_spin_operators(n: int) -> Tuple[Array, Array, Array]:
    """
    Build Jx, Jy, Jz in the symmetric Dicke basis |D_k^n>, k = 0,...,n.

    Here k is the number of |1> excitations. Since Z|0>=|0> and Z|1>=-|1>,
    Jz |D_k^n> = (n/2 - k) |D_k^n>.
    """
    dim = n + 1
    jp = np.zeros((dim, dim), dtype=complex)
    jm = np.zeros((dim, dim), dtype=complex)

    for k in range(1, dim):
        jp[k - 1, k] = math.sqrt(k * (n - k + 1))
    for k in range(dim - 1):
        jm[k + 1, k] = math.sqrt((k + 1) * (n - k))

    jx = 0.5 * (jp + jm)
    jy = (jp - jm) / (2.0j)
    m_vals = np.array([n / 2.0 - k for k in range(dim)], dtype=float)
    jz = np.diag(m_vals).astype(complex)
    return jx, jy, jz


def initial_symmetric_state(n: int) -> Array:
    psi = np.zeros(n + 1, dtype=complex)
    psi[0] = 1.0
    return psi


def lie_forward(theta: Array, n: int, lie_layers: int, ops: Tuple[Array, Array, Array]) -> Array:
    """
    Collective-spin Lie ansatz:

        prod_l exp(-i alpha_l Jx) exp(-i beta_l Jz^2 / n) exp(-i gamma_l Jy).
    """
    jx, jy, jz = ops
    jz_diag = np.diag(jz).real
    params = theta.reshape(lie_layers, 3)
    psi = initial_symmetric_state(n)

    for alpha, beta, gamma in params:
        psi = expm(-1j * alpha * jx) @ psi
        psi = np.exp(-1j * beta * (jz_diag**2) / max(n, 1)) * psi
        psi = expm(-1j * gamma * jy) @ psi
    return psi


def lie_outputs(theta: Array, n: int, lie_layers: int, ops: Tuple[Array, Array, Array]) -> Array:
    """Return normalized collective observables (O_z, O_x)."""
    jx, _, jz = ops
    psi = lie_forward(theta, n, lie_layers, ops)
    oz = (2.0 / n) * np.vdot(psi, jz @ psi).real
    ox = (2.0 / n) * np.vdot(psi, jx @ psi).real
    return np.array([oz, ox], dtype=float)


def sample_angles(
    rng: np.random.Generator,
    size: int,
    mode: str,
    normal_sigma: float,
) -> Array:
    if mode == "uniform":
        return rng.uniform(-math.pi, math.pi, size=size)
    if mode == "normal":
        return rng.normal(0.0, normal_sigma, size=size)
    if mode == "normal_inv_sqrt_dim":
        return rng.normal(0.0, 1.0 / math.sqrt(size), size=size)
    raise ValueError(f"Unknown angle mode: {mode}")


def make_teacher_theta(
    rng: np.random.Generator,
    lie_param_count: int,
    teacher_mode: str,
    teacher_scale: float,
) -> Array:
    theta = np.zeros(lie_param_count, dtype=float)

    if teacher_mode == "deterministic_margin":
        # First layer gamma = pi/2. This sends the all-zero coherent spin state
        # from +z toward +x, producing an O(1) target separation for every n.
        theta[2] = math.pi / 2.0
        return theta

    if teacher_mode == "deterministic_z_flip":
        # First layer gamma = pi. This maps O_z from +1 to approximately -1.
        theta[2] = math.pi
        return theta

    if teacher_mode == "random_normal":
        return rng.normal(0.0, teacher_scale, size=lie_param_count)

    raise ValueError(f"Unknown teacher mode: {teacher_mode}")


def run_single_n(
    n: int,
    lie_layers: int,
    theta_samples: int,
    angle_mode: str,
    normal_sigma: float,
    teacher_mode: str,
    teacher_scale: float,
    rng: np.random.Generator,
    progress_every: int,
) -> ResidualResult:
    ops = build_collective_spin_operators(n)
    lie_param_count = 3 * lie_layers

    theta_teacher = make_teacher_theta(rng, lie_param_count, teacher_mode, teacher_scale)
    y_target = lie_outputs(theta_teacher, n, lie_layers, ops)

    theta_zero = np.zeros(lie_param_count, dtype=float)
    y_zero = lie_outputs(theta_zero, n, lie_layers, ops)
    zero_residual_sq = float(np.dot(y_zero - y_target, y_zero - y_target))

    vector_residual_sq = np.empty(theta_samples, dtype=float)
    oz_residual_sq = np.empty(theta_samples, dtype=float)
    ox_residual_sq = np.empty(theta_samples, dtype=float)

    start = time.perf_counter()
    for s in range(theta_samples):
        theta = sample_angles(rng, lie_param_count, angle_mode, normal_sigma)
        y_pred = lie_outputs(theta, n, lie_layers, ops)
        residual = y_pred - y_target

        oz_residual_sq[s] = residual[0] ** 2
        ox_residual_sq[s] = residual[1] ** 2
        vector_residual_sq[s] = float(np.dot(residual, residual))

        done = s + 1
        if done == 1 or done == theta_samples or done % progress_every == 0:
            elapsed = time.perf_counter() - start
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = (theta_samples - done) / rate if rate > 0 else float("inf")
            print(
                f"[n={n}] {done}/{theta_samples} samples, "
                f"elapsed {elapsed:.1f}s, eta {format_eta(eta)}",
                flush=True,
            )

    elapsed = time.perf_counter() - start
    mean_oz = float(np.mean(oz_residual_sq))
    mean_ox = float(np.mean(ox_residual_sq))
    if mean_oz >= mean_ox:
        witness_observable = "Oz"
        witness_beta = mean_oz
    else:
        witness_observable = "Ox"
        witness_beta = mean_ox

    return ResidualResult(
        n=n,
        symmetric_dim=n + 1,
        lie_params=lie_param_count,
        theta_samples=theta_samples,
        target_oz=float(y_target[0]),
        target_ox=float(y_target[1]),
        zero_oz=float(y_zero[0]),
        zero_ox=float(y_zero[1]),
        zero_residual_sq=zero_residual_sq,
        mean_vector_residual_sq=float(np.mean(vector_residual_sq)),
        std_vector_residual_sq=float(np.std(vector_residual_sq)),
        median_vector_residual_sq=float(np.median(vector_residual_sq)),
        q05_vector_residual_sq=float(np.quantile(vector_residual_sq, 0.05)),
        q95_vector_residual_sq=float(np.quantile(vector_residual_sq, 0.95)),
        min_vector_residual_sq=float(np.min(vector_residual_sq)),
        max_vector_residual_sq=float(np.max(vector_residual_sq)),
        mean_oz_residual_sq=mean_oz,
        mean_ox_residual_sq=mean_ox,
        witness_observable=witness_observable,
        witness_beta_estimate=witness_beta,
        elapsed_sec=elapsed,
    )


def save_results_csv(path: Path, results: Sequence[ResidualResult]) -> None:
    fieldnames = list(asdict(results[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def plot_results(path: Path, results: Sequence[ResidualResult]) -> None:
    ns = np.array([r.n for r in results], dtype=float)
    mean_vec = np.array([r.mean_vector_residual_sq for r in results], dtype=float)
    mean_oz = np.array([r.mean_oz_residual_sq for r in results], dtype=float)
    mean_ox = np.array([r.mean_ox_residual_sq for r in results], dtype=float)
    witness = np.array([r.witness_beta_estimate for r in results], dtype=float)
    zero_res = np.array([r.zero_residual_sq for r in results], dtype=float)

    beta_floor = float(np.min(witness))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].plot(ns, mean_vec, marker="o", label=r"$E_\theta\|f(\theta)-y\|^2$")
    axes[0].plot(ns, mean_oz, marker="s", label=r"$E_\theta[(O_z-y_z)^2]$")
    axes[0].plot(ns, mean_ox, marker="^", label=r"$E_\theta[(O_x-y_x)^2]$")
    axes[0].plot(ns, zero_res, marker="x", linestyle="--", label="zero-parameter residual")
    axes[0].set_xlabel("n")
    axes[0].set_ylabel("Mean squared residual")
    axes[0].set_title("Initial residual scaling")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(ns, witness, marker="o", color="tab:purple", label="witness component")
    axes[1].axhline(
        beta_floor,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=rf"min observed $\beta$ = {beta_floor:.3g}",
    )
    axes[1].set_xlabel("n")
    axes[1].set_ylabel("Witness beta estimate")
    axes[1].set_title("Empirical n-independent lower bound")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_report(results: Sequence[ResidualResult]) -> None:
    header = (
        "n  dim  params  target(Oz,Ox)        zeroRes     meanVec     "
        "meanOz      meanOx      witness beta"
    )
    print()
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.n:<2d} "
            f"{r.symmetric_dim:<4d} "
            f"{r.lie_params:<6d} "
            f"({r.target_oz:+.4f},{r.target_ox:+.4f})   "
            f"{r.zero_residual_sq:>9.4e} "
            f"{r.mean_vector_residual_sq:>9.4e} "
            f"{r.mean_oz_residual_sq:>9.4e} "
            f"{r.mean_ox_residual_sq:>9.4e} "
            f"{r.witness_observable:>2s}={r.witness_beta_estimate:.4e}"
        )

    beta_floor = min(r.witness_beta_estimate for r in results)
    vec_floor = min(r.mean_vector_residual_sq for r in results)
    print()
    print(f"Minimum witness beta over tested n: {beta_floor:.6e}")
    print(f"Minimum vector residual over tested n: {vec_floor:.6e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initial residual scaling experiment for collective-spin Lie-QNN."
    )
    parser.add_argument(
        "--n-values",
        nargs="+",
        type=int,
        default=[3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32],
        help="System sizes to evaluate.",
    )
    parser.add_argument(
        "--theta-samples",
        type=int,
        default=1000,
        help="Number of random initial theta samples per n.",
    )
    parser.add_argument(
        "--lie-layers",
        type=int,
        default=2,
        help="Number of collective-spin Lie-QNN layers.",
    )
    parser.add_argument(
        "--angle-mode",
        choices=["uniform", "normal", "normal_inv_sqrt_dim"],
        default="uniform",
        help="Random initialization distribution for trainable parameters.",
    )
    parser.add_argument(
        "--normal-sigma",
        type=float,
        default=1.0,
        help="Sigma used when --angle-mode normal.",
    )
    parser.add_argument(
        "--teacher-mode",
        choices=["deterministic_margin", "deterministic_z_flip", "random_normal"],
        default="deterministic_margin",
        help="How to generate the Lie teacher target.",
    )
    parser.add_argument(
        "--teacher-scale",
        type=float,
        default=1.0,
        help="Normal scale for teacher parameters when --teacher-mode random_normal.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Master random seed.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("initial_residual_scaling_results.csv"),
        help="CSV output path.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("initial_residual_scaling_summary.png"),
        help="Plot output path.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plot generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.lie_layers < 1:
        raise ValueError("--lie-layers must be at least 1.")
    if args.theta_samples < 1:
        raise ValueError("--theta-samples must be at least 1.")

    print(
        "Starting initial-residual scaling experiment with "
        f"n={args.n_values}, theta_samples={args.theta_samples}, "
        f"lie_layers={args.lie_layers}, angle_mode={args.angle_mode}, "
        f"teacher_mode={args.teacher_mode}",
        flush=True,
    )

    master_rng = np.random.default_rng(args.seed)
    results: List[ResidualResult] = []

    for n in args.n_values:
        print(f"\nRunning system size n={n}", flush=True)
        case_rng = np.random.default_rng(master_rng.integers(0, 2**32 - 1))
        result = run_single_n(
            n=n,
            lie_layers=args.lie_layers,
            theta_samples=args.theta_samples,
            angle_mode=args.angle_mode,
            normal_sigma=args.normal_sigma,
            teacher_mode=args.teacher_mode,
            teacher_scale=args.teacher_scale,
            rng=case_rng,
            progress_every=args.progress_every,
        )
        results.append(result)
        print(
            f"Finished n={n}: mean residual={result.mean_vector_residual_sq:.4e}, "
            f"witness {result.witness_observable} beta="
            f"{result.witness_beta_estimate:.4e}, elapsed={result.elapsed_sec:.1f}s",
            flush=True,
        )

    print_report(results)
    save_results_csv(args.csv, results)
    if not args.no_plot:
        plot_results(args.plot, results)

    print(f"\nSaved CSV to {args.csv}", flush=True)
    if not args.no_plot:
        print(f"Saved plot to {args.plot}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
