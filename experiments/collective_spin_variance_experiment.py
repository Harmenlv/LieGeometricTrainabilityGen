#!/usr/bin/env python3
"""
Collective-spin Lie-QNN variance experiment.

This script is a replacement template for the appendix-style variance tests in
Lie_Geometric_Trainability.ipynb.  It keeps the paper's basic trainability
protocol:

  - fix the initial state and observables for each system size n
  - generate synthetic targets from the Lie-restricted model
  - sample only trainable parameters theta
  - compare gradient variance between a Lie-restricted model and a full-Hilbert
    hardware-efficient QNN baseline

The Lie branch uses the permutation-symmetric collective-spin ansatz

    U_Lie(theta) = prod_l exp(-i a_l Jx) exp(-i b_l Jz^2 / n) exp(-i c_l Jy),

where Jx, Jy, Jz act on the symmetric Dicke subspace of dimension n + 1.
This makes simulations for n >= 10 cheap and gives a model whose representation
dimension is polynomial in n.

The full branch is an ordinary statevector hardware-efficient QNN on the full
2^n-dimensional Hilbert space:

    [RY_i(theta) RZ_i(phi) for all qubits] + nearest-neighbor CZ entanglers.

It is not an exact Haar SU(2^n) sampler.  Use larger --full-layers if you want
the full baseline to be more scrambling and closer to the barren-plateau
intuition in Section VII.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm


Array = np.ndarray


@dataclass
class VarianceResult:
    n: int
    lie_subspace_dim: int
    lie_params: int
    full_hilbert_dim: int
    full_params: int
    full_grad_params: int
    mean_lie: float
    max_lie: float
    min_lie: float
    mean_full: float
    max_full: float
    min_full: float
    ratio_mean: float
    ratio_max: float
    ratio_min: float
    section_vii_heuristic_ratio: float
    elapsed_lie_sec: float
    elapsed_full_sec: float


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

    Here k is the number of |1> excitations.  Since Z|0>=|0> and Z|1>=-|1>,
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
    """Return final symmetric-subspace state for the collective-spin Lie ansatz."""
    jx, jy, jz = ops
    jz_diag = np.diag(jz).real
    psi = initial_symmetric_state(n)
    params = theta.reshape(lie_layers, 3)

    for alpha, beta, gamma in params:
        psi = expm(-1j * alpha * jx) @ psi
        psi = np.exp(-1j * beta * (jz_diag**2) / max(n, 1)) * psi
        psi = expm(-1j * gamma * jy) @ psi
    return psi


def lie_outputs(theta: Array, n: int, lie_layers: int, ops: Tuple[Array, Array, Array]) -> Array:
    """
    Normalized collective observables.

    O_z = 2 Jz / n and O_x = 2 Jx / n both have O(1) scale as n grows.
    """
    jx, _, jz = ops
    psi = lie_forward(theta, n, lie_layers, ops)
    oz = (2.0 / n) * np.vdot(psi, jz @ psi).real
    ox = (2.0 / n) * np.vdot(psi, jx @ psi).real
    return np.array([oz, ox], dtype=float)


def lie_loss_and_grad(
    theta: Array,
    n: int,
    lie_layers: int,
    ops: Tuple[Array, Array, Array],
    y_target: Array,
    eps: float,
) -> Tuple[float, Array]:
    """
    Central-difference gradient for the collective-spin Lie ansatz.

    Jz^2 has more than two eigenvalues, so the simple two-point parameter-shift
    formula is not generally exact for all Lie gates.  Since the symmetric
    subspace dimension is only n+1, central differences are cheap here.
    """
    base = lie_outputs(theta, n, lie_layers, ops)
    diff = base - y_target
    loss = float(np.dot(diff, diff))
    grad = np.zeros_like(theta, dtype=float)

    for k in range(theta.size):
        plus = theta.copy()
        minus = theta.copy()
        plus[k] += eps
        minus[k] -= eps
        dy = (lie_outputs(plus, n, lie_layers, ops) - lie_outputs(minus, n, lie_layers, ops)) / (
            2.0 * eps
        )
        grad[k] = 2.0 * float(np.dot(diff, dy))
    return loss, grad


def ry(theta: float) -> Array:
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(theta: float) -> Array:
    return np.array(
        [[np.exp(-0.5j * theta), 0.0], [0.0, np.exp(0.5j * theta)]],
        dtype=complex,
    )


def apply_single_qubit_gate(state: Array, gate: Array, qubit: int, n: int) -> Array:
    """Apply a one-qubit gate.  Qubit 0 is the leftmost tensor axis."""
    tensor = state.reshape((2,) * n)
    tensor = np.moveaxis(tensor, qubit, 0)
    updated = np.tensordot(gate, tensor, axes=([1], [0]))
    updated = np.moveaxis(updated, 0, qubit)
    return updated.reshape(-1)


def build_cz_phase(n: int, ring: bool = False) -> Array:
    dim = 2**n
    phase = np.ones(dim, dtype=complex)
    pairs = [(q, q + 1) for q in range(n - 1)]
    if ring and n > 2:
        pairs.append((n - 1, 0))

    indices = np.arange(dim)
    for q1, q2 in pairs:
        mask1 = 1 << (n - 1 - q1)
        mask2 = 1 << (n - 1 - q2)
        both_one = ((indices & mask1) != 0) & ((indices & mask2) != 0)
        phase[both_one] *= -1.0
    return phase


def initial_full_state(n: int) -> Array:
    psi = np.zeros(2**n, dtype=complex)
    psi[0] = 1.0
    return psi


def full_forward(theta: Array, n: int, full_layers: int, cz_phase: Array) -> Array:
    """Return final state for the full-Hilbert hardware-efficient QNN."""
    params = theta.reshape(full_layers, n, 2)
    psi = initial_full_state(n)
    for layer in range(full_layers):
        for q in range(n):
            psi = apply_single_qubit_gate(psi, ry(params[layer, q, 0]), q, n)
            psi = apply_single_qubit_gate(psi, rz(params[layer, q, 1]), q, n)
        psi = cz_phase * psi
    return psi


def precompute_full_observable_tables(n: int) -> Tuple[Array, List[Array]]:
    dim = 2**n
    indices = np.arange(dim)

    z_mean = np.zeros(dim, dtype=float)
    flip_indices: List[Array] = []
    for q in range(n):
        mask = 1 << (n - 1 - q)
        z_q = np.where((indices & mask) == 0, 1.0, -1.0)
        z_mean += z_q / n
        flip_indices.append(indices ^ mask)
    return z_mean, flip_indices


def full_outputs(
    theta: Array,
    n: int,
    full_layers: int,
    cz_phase: Array,
    obs_tables: Tuple[Array, List[Array]],
) -> Array:
    """Return the same normalized collective observables used by the Lie branch."""
    psi = full_forward(theta, n, full_layers, cz_phase)
    probs = np.abs(psi) ** 2
    z_mean, flip_indices = obs_tables

    oz = float(np.dot(probs, z_mean))
    ox = 0.0
    for flipped in flip_indices:
        ox += np.vdot(psi, psi[flipped]).real / n
    return np.array([oz, ox], dtype=float)


def full_loss_and_grad_subset(
    theta: Array,
    n: int,
    full_layers: int,
    cz_phase: Array,
    obs_tables: Tuple[Array, List[Array]],
    y_target: Array,
    grad_indices: Sequence[int],
) -> Tuple[float, Array]:
    """
    Parameter-shift gradient for selected RY/RZ parameters of the full QNN.

    For gates exp(-i theta P/2), dy/dtheta = [y(theta+pi/2)-y(theta-pi/2)]/2.
    """
    base = full_outputs(theta, n, full_layers, cz_phase, obs_tables)
    diff = base - y_target
    loss = float(np.dot(diff, diff))
    grad = np.zeros(len(grad_indices), dtype=float)

    shift = math.pi / 2.0
    for out_idx, flat_idx in enumerate(grad_indices):
        plus = theta.copy()
        minus = theta.copy()
        plus[flat_idx] += shift
        minus[flat_idx] -= shift
        dy = (
            full_outputs(plus, n, full_layers, cz_phase, obs_tables)
            - full_outputs(minus, n, full_layers, cz_phase, obs_tables)
        ) / 2.0
        grad[out_idx] = 2.0 * float(np.dot(diff, dy))
    return loss, grad


def sample_angles(rng: np.random.Generator, size: int, mode: str, sigma: float) -> Array:
    if mode == "uniform":
        return rng.uniform(-math.pi, math.pi, size=size)
    if mode == "normal":
        return rng.normal(0.0, sigma, size=size)
    if mode == "normal_inv_sqrt_dim":
        return rng.normal(0.0, 1.0 / math.sqrt(size), size=size)
    raise ValueError(f"Unknown angle sampling mode: {mode}")


def estimate_lie_variance(
    n: int,
    lie_layers: int,
    ops: Tuple[Array, Array, Array],
    y_target: Array,
    rng: np.random.Generator,
    theta_samples: int,
    angle_mode: str,
    normal_sigma: float,
    finite_diff_eps: float,
    progress_every: int,
) -> Tuple[float, float, float, Array, float]:
    param_count = 3 * lie_layers
    grads = np.empty((theta_samples, param_count), dtype=float)
    start = time.perf_counter()

    for s in range(theta_samples):
        theta = sample_angles(rng, param_count, angle_mode, normal_sigma)
        _, grad = lie_loss_and_grad(theta, n, lie_layers, ops, y_target, finite_diff_eps)
        grads[s] = grad

        done = s + 1
        if done == 1 or done == theta_samples or done % progress_every == 0:
            elapsed = time.perf_counter() - start
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = (theta_samples - done) / rate if rate > 0 else float("inf")
            print(
                f"[n={n} Lie] {done}/{theta_samples} samples, "
                f"elapsed {elapsed:.1f}s, eta {format_eta(eta)}",
                flush=True,
            )

    elapsed = time.perf_counter() - start
    var_per_param = np.var(grads, axis=0)
    return (
        float(np.mean(var_per_param)),
        float(np.max(var_per_param)),
        float(np.min(var_per_param)),
        var_per_param,
        elapsed,
    )


def estimate_full_variance(
    n: int,
    full_layers: int,
    y_target: Array,
    rng: np.random.Generator,
    theta_samples: int,
    angle_mode: str,
    normal_sigma: float,
    full_grad_params: int,
    progress_every: int,
    ring: bool,
) -> Tuple[float, float, float, Array, int, int, float]:
    full_param_count = 2 * n * full_layers
    if full_grad_params <= 0 or full_grad_params >= full_param_count:
        grad_indices = np.arange(full_param_count)
    else:
        grad_indices = np.sort(rng.choice(full_param_count, size=full_grad_params, replace=False))

    cz_phase = build_cz_phase(n, ring=ring)
    obs_tables = precompute_full_observable_tables(n)
    grads = np.empty((theta_samples, len(grad_indices)), dtype=float)
    start = time.perf_counter()

    for s in range(theta_samples):
        theta = sample_angles(rng, full_param_count, angle_mode, normal_sigma)
        _, grad = full_loss_and_grad_subset(
            theta,
            n,
            full_layers,
            cz_phase,
            obs_tables,
            y_target,
            grad_indices,
        )
        grads[s] = grad

        done = s + 1
        if done == 1 or done == theta_samples or done % progress_every == 0:
            elapsed = time.perf_counter() - start
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = (theta_samples - done) / rate if rate > 0 else float("inf")
            print(
                f"[n={n} Full] {done}/{theta_samples} samples, "
                f"elapsed {elapsed:.1f}s, eta {format_eta(eta)}",
                flush=True,
            )

    elapsed = time.perf_counter() - start
    var_per_param = np.var(grads, axis=0)
    return (
        float(np.mean(var_per_param)),
        float(np.max(var_per_param)),
        float(np.min(var_per_param)),
        var_per_param,
        full_param_count,
        len(grad_indices),
        elapsed,
    )


def run_single_n(
    n: int,
    lie_layers: int,
    full_layers: int,
    theta_samples: int,
    angle_mode: str,
    normal_sigma: float,
    teacher_scale: float,
    finite_diff_eps: float,
    full_grad_params: int,
    rng: np.random.Generator,
    progress_every: int,
    ring: bool,
) -> VarianceResult:
    ops = build_collective_spin_operators(n)
    lie_param_count = 3 * lie_layers

    # Synthetic target generated by the Lie-restricted teacher.
    theta_teacher = rng.normal(0.0, teacher_scale, size=lie_param_count)
    y_target = lie_outputs(theta_teacher, n, lie_layers, ops)
    print(
        f"n={n}: target observables from Lie teacher "
        f"[Oz={y_target[0]:+.5f}, Ox={y_target[1]:+.5f}]",
        flush=True,
    )

    lie_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
    full_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))

    mean_l, max_l, min_l, _, elapsed_l = estimate_lie_variance(
        n=n,
        lie_layers=lie_layers,
        ops=ops,
        y_target=y_target,
        rng=lie_rng,
        theta_samples=theta_samples,
        angle_mode=angle_mode,
        normal_sigma=normal_sigma,
        finite_diff_eps=finite_diff_eps,
        progress_every=progress_every,
    )

    mean_f, max_f, min_f, _, full_param_count, used_full_grad_params, elapsed_f = (
        estimate_full_variance(
            n=n,
            full_layers=full_layers,
            y_target=y_target,
            rng=full_rng,
            theta_samples=theta_samples,
            angle_mode=angle_mode,
            normal_sigma=normal_sigma,
            full_grad_params=full_grad_params,
            progress_every=progress_every,
            ring=ring,
        )
    )

    ratio_mean = mean_l / mean_f if mean_f != 0 else float("inf")
    ratio_max = max_l / max_f if max_f != 0 else float("inf")
    ratio_min = min_l / min_f if min_f != 0 else float("inf")

    # Section VII-style heuristic after replacing the old 2n-1 generator count
    # by the collective-spin irrep dimension n+1.
    heuristic_ratio = (2.0**n) / (n + 1.0)

    return VarianceResult(
        n=n,
        lie_subspace_dim=n + 1,
        lie_params=lie_param_count,
        full_hilbert_dim=2**n,
        full_params=full_param_count,
        full_grad_params=used_full_grad_params,
        mean_lie=mean_l,
        max_lie=max_l,
        min_lie=min_l,
        mean_full=mean_f,
        max_full=max_f,
        min_full=min_f,
        ratio_mean=ratio_mean,
        ratio_max=ratio_max,
        ratio_min=ratio_min,
        section_vii_heuristic_ratio=heuristic_ratio,
        elapsed_lie_sec=elapsed_l,
        elapsed_full_sec=elapsed_f,
    )


def save_csv(path: Path, results: Sequence[VarianceResult]) -> None:
    if not results:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for item in results:
            writer.writerow(asdict(item))


def plot_summary(path: Path, results: Sequence[VarianceResult]) -> None:
    ns = np.array([r.n for r in results], dtype=float)
    lie = np.array([r.mean_lie for r in results], dtype=float)
    full = np.array([r.mean_full for r in results], dtype=float)
    ratio = np.array([r.ratio_mean for r in results], dtype=float)
    heuristic = np.array([r.section_vii_heuristic_ratio for r in results], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))

    axes[0].plot(ns, lie, "o-", color="tab:blue", label="Collective Lie QNN")
    axes[0].plot(ns, full, "o-", color="tab:red", label="Full-Hilbert QNN")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("n")
    axes[0].set_ylabel("Mean gradient variance")
    axes[0].set_title("Gradient variance")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(ns, ratio, "o-", color="tab:purple", label="Observed Lie / Full")
    axes[1].plot(ns, heuristic, "--", color="black", label=r"Heuristic $2^n/(n+1)$")
    axes[1].axhline(1.0, color="gray", linestyle=":", linewidth=1.2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("n")
    axes[1].set_ylabel("Variance ratio")
    axes[1].set_title("Ratio")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_report(results: Sequence[VarianceResult]) -> None:
    header = (
        "n  lieDim fullDim lieP fullP usedFullP  "
        "meanLie      meanFull     ratioMean   heuristic"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.n:<2d} "
            f"{r.lie_subspace_dim:<6d} "
            f"{r.full_hilbert_dim:<7d} "
            f"{r.lie_params:<4d} "
            f"{r.full_params:<5d} "
            f"{r.full_grad_params:<9d} "
            f"{r.mean_lie:>11.4e} "
            f"{r.mean_full:>11.4e} "
            f"{r.ratio_mean:>11.4f} "
            f"{r.section_vii_heuristic_ratio:>10.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collective-spin Lie-QNN vs full-Hilbert QNN variance experiment."
    )
    parser.add_argument(
        "--n-values",
        nargs="+",
        type=int,
        default=[3, 4, 5, 6, 8, 10, 12],
        help="System sizes to evaluate.",
    )
    parser.add_argument(
        "--theta-samples",
        type=int,
        default=500,
        help="Number of random theta samples per model and per n.",
    )
    parser.add_argument(
        "--lie-layers",
        type=int,
        default=2,
        help="Number of collective-spin Lie ansatz layers.",
    )
    parser.add_argument(
        "--full-layers",
        type=int,
        default=8,
        help=(
            "Number of hardware-efficient full-QNN layers. "
            "Use larger values, e.g. 10 or 2*n, for stronger scrambling."
        ),
    )
    parser.add_argument(
        "--full-grad-params",
        type=int,
        default=0,
        help=(
            "Number of full-QNN parameters whose gradients are estimated. "
            "Use 0 to evaluate all full parameters."
        ),
    )
    parser.add_argument(
        "--angle-mode",
        choices=["uniform", "normal", "normal_inv_sqrt_dim"],
        default="uniform",
        help="Parameter initialization distribution.",
    )
    parser.add_argument(
        "--normal-sigma",
        type=float,
        default=1.0,
        help="Sigma used when --angle-mode normal.",
    )
    parser.add_argument(
        "--teacher-scale",
        type=float,
        default=1.0,
        help="Normal scale for the Lie teacher parameters that generate y_target.",
    )
    parser.add_argument(
        "--finite-diff-eps",
        type=float,
        default=1.0e-5,
        help="Central-difference epsilon for Lie gradients.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master random seed.",
    )
    parser.add_argument(
        "--ring",
        action="store_true",
        help="Use ring CZ entanglers in the full-QNN baseline.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("collective_spin_variance_results.csv"),
        help="CSV output path.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("collective_spin_variance_summary.png"),
        help="Summary plot output path.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plot generation.",
    )
    args = parser.parse_args()

    if min(args.n_values) < 2:
        raise ValueError("Use n >= 2 for this collective-spin experiment.")

    print(
        "Starting collective-spin variance experiment with "
        f"n={args.n_values}, theta_samples={args.theta_samples}, "
        f"lie_layers={args.lie_layers}, full_layers={args.full_layers}, "
        f"angle_mode={args.angle_mode}",
        flush=True,
    )

    master_rng = np.random.default_rng(args.seed)
    results: List[VarianceResult] = []

    for n in args.n_values:
        print(f"\nRunning system size n={n}", flush=True)
        case_rng = np.random.default_rng(master_rng.integers(0, 2**32 - 1))
        result = run_single_n(
            n=n,
            lie_layers=args.lie_layers,
            full_layers=args.full_layers,
            theta_samples=args.theta_samples,
            angle_mode=args.angle_mode,
            normal_sigma=args.normal_sigma,
            teacher_scale=args.teacher_scale,
            finite_diff_eps=args.finite_diff_eps,
            full_grad_params=args.full_grad_params,
            rng=case_rng,
            progress_every=args.progress_every,
            ring=args.ring,
        )
        results.append(result)
        print(
            f"Finished n={n}: mean ratio={result.ratio_mean:.4f}, "
            f"Lie time={result.elapsed_lie_sec:.1f}s, "
            f"Full time={result.elapsed_full_sec:.1f}s",
            flush=True,
        )

    print()
    print_report(results)
    save_csv(args.csv, results)
    if not args.no_plot:
        plot_summary(args.plot, results)

    print(f"\nSaved CSV to {args.csv}", flush=True)
    if not args.no_plot:
        print(f"Saved plot to {args.plot}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
