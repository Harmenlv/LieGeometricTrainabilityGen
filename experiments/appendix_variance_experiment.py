#!/usr/bin/env python3
"""
Appendix variance experiment for Lie-restricted vs full parameterizations.

This script follows the source notebook's protocol:
  - fix one (rho0, O) pair per system size n
  - generate synthetic targets from a restricted Hamiltonian
  - sample only theta during variance estimation
  - compare Lie-restricted and full Pauli parameterizations

The gradient rule matches the notebook's commutator-based convention.
It is intended as a reproducible experimental template for n = 3, 4, 5.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm


PAULI_MATS: Dict[str, np.ndarray] = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


@dataclass
class VarianceResult:
    n: int
    dim_restricted: int
    dim_full: int
    mean_restricted: float
    max_restricted: float
    min_restricted: float
    mean_full: float
    max_full: float
    min_full: float
    ratio_mean: float
    ratio_max: float
    ratio_min: float


def dagger(a: np.ndarray) -> np.ndarray:
    return a.conj().T


def comm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b - b @ a


def kron_all(ops: Sequence[np.ndarray]) -> np.ndarray:
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def build_all_pauli_basis(n: int) -> Tuple[List[str], List[np.ndarray]]:
    labels: List[str] = []
    mats: List[np.ndarray] = []
    for word in itertools.product("IXYZ", repeat=n):
        ops = [PAULI_MATS[s] for s in word]
        labels.append("".join(word))
        mats.append(kron_all(ops))
    return labels, mats


def build_restricted_generators(n: int) -> Tuple[List[str], List[np.ndarray]]:
    labels: List[str] = []
    mats: List[np.ndarray] = []

    # Local X_i terms.
    for i in range(n):
        ops = [PAULI_MATS["I"]] * n
        ops[i] = PAULI_MATS["X"]
        labels.append(f"X{i + 1}")
        mats.append(kron_all(ops))

    # Nearest-neighbor Z_i Z_{i+1} terms.
    for i in range(n - 1):
        ops = [PAULI_MATS["I"]] * n
        ops[i] = PAULI_MATS["Z"]
        ops[i + 1] = PAULI_MATS["Z"]
        labels.append(f"Z{i + 1}Z{i + 2}")
        mats.append(kron_all(ops))

    return labels, mats


def build_full_generators(n: int) -> Tuple[List[str], List[np.ndarray]]:
    all_labels, all_mats = build_all_pauli_basis(n)
    identity_label = "I" * n
    labels = []
    mats = []
    for label, mat in zip(all_labels, all_mats):
        if label == identity_label:
            continue
        labels.append(label)
        mats.append(mat)
    return labels, mats


def random_density_matrix(dim: int, rng: np.random.Generator) -> np.ndarray:
    a = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    rho = a @ dagger(a)
    rho /= np.trace(rho)
    return rho


def random_observable_from_pauli_basis(
    basis_mats: Sequence[np.ndarray], rng: np.random.Generator
) -> np.ndarray:
    coeffs = rng.uniform(-1.0, 1.0, size=len(basis_mats))
    observable = np.zeros_like(basis_mats[0], dtype=complex)
    for coeff, mat in zip(coeffs, basis_mats):
        observable = observable + coeff * mat
    observable = 0.5 * (observable + dagger(observable))
    fro_norm = np.linalg.norm(observable, ord="fro")
    if fro_norm == 0.0:
        raise RuntimeError("Random observable collapsed to zero norm.")
    return observable / fro_norm


def hamiltonian_from_theta(theta: np.ndarray, generators: Sequence[np.ndarray]) -> np.ndarray:
    h = np.zeros_like(generators[0], dtype=complex)
    for coeff, gen in zip(theta, generators):
        h = h + coeff * gen
    return h


def quantum_flow(h: np.ndarray, rho: np.ndarray, t: float) -> np.ndarray:
    u = expm(-1j * t * h)
    return u @ rho @ dagger(u)


def loss_and_grad(
    theta: np.ndarray,
    generators: Sequence[np.ndarray],
    rho0: np.ndarray,
    observable: np.ndarray,
    t_list: Sequence[float],
    y_data: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    Notebook-style loss and gradient.

    The gradient rule mirrors the source notebook:
        d rho_t / d theta_k ~= -i t [U X_k U^dagger, rho_t]
    """
    h = hamiltonian_from_theta(theta, generators)
    grad = np.zeros_like(theta, dtype=float)
    loss = 0.0

    for idx, t in enumerate(t_list):
        rho_t = quantum_flow(h, rho0, t)
        y = np.trace(observable @ rho_t).real
        diff = y - y_data[idx]
        loss += diff * diff

        u = expm(-1j * t * h)
        for k, xk in enumerate(generators):
            xk_t = u @ xk @ dagger(u)
            drho = -1j * t * comm(xk_t, rho_t)
            dy = np.trace(observable @ drho).real
            grad[k] += 2.0 * diff * dy

    return loss, grad


def gradient_for_theta(
    theta: np.ndarray,
    generators: Sequence[np.ndarray],
    rho0: np.ndarray,
    observable: np.ndarray,
    t_list: Sequence[float],
    y_data: np.ndarray,
) -> np.ndarray:
    _, grad = loss_and_grad(theta, generators, rho0, observable, t_list, y_data)
    return grad


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


def gradient_variance_fixed_rho_o(
    generators: Sequence[np.ndarray],
    rho0: np.ndarray,
    observable: np.ndarray,
    y_data: np.ndarray,
    t_list: Sequence[float],
    rng: np.random.Generator,
    n_theta_samples: int = 1000,
    workers: int = 1,
    progress_label: str = "",
    progress_every: int = 50,
) -> Tuple[float, float, float, np.ndarray]:
    """
    Fix (rho0, O), sample theta only, and estimate variance per parameter.
    Returns mean, max, min variance and the full per-parameter variance array.
    """
    d = len(generators)
    sigma = 1.0 / np.sqrt(d)

    grads = np.empty((n_theta_samples, d), dtype=float)
    start = time.perf_counter()

    if workers <= 1:
        for s in range(n_theta_samples):
            theta = rng.normal(loc=0.0, scale=sigma, size=d)
            grads[s] = gradient_for_theta(theta, generators, rho0, observable, t_list, y_data)

            done = s + 1
            if done == 1 or done == n_theta_samples or done % progress_every == 0:
                elapsed = time.perf_counter() - start
                rate = done / elapsed if elapsed > 0 else 0.0
                eta = (n_theta_samples - done) / rate if rate > 0 else float("inf")
                prefix = f"[{progress_label}] " if progress_label else ""
                print(
                    f"{prefix}{done}/{n_theta_samples} samples, "
                    f"elapsed {elapsed:.1f}s, eta {format_eta(eta)}",
                    flush=True,
                )
    else:
        thetas = [rng.normal(loc=0.0, scale=sigma, size=d) for _ in range(n_theta_samples)]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    gradient_for_theta,
                    theta,
                    generators,
                    rho0,
                    observable,
                    t_list,
                    y_data,
                ): idx
                for idx, theta in enumerate(thetas)
            }
            done = 0
            for future in as_completed(futures):
                idx = futures[future]
                grads[idx] = future.result()
                done += 1
                if done == 1 or done == n_theta_samples or done % progress_every == 0:
                    elapsed = time.perf_counter() - start
                    rate = done / elapsed if elapsed > 0 else 0.0
                    eta = (n_theta_samples - done) / rate if rate > 0 else float("inf")
                    prefix = f"[{progress_label}] " if progress_label else ""
                    print(
                        f"{prefix}{done}/{n_theta_samples} samples, "
                        f"elapsed {elapsed:.1f}s, eta {format_eta(eta)}",
                        flush=True,
                    )

    var_per_param = np.var(grads, axis=0)
    mean_var = float(np.mean(var_per_param))
    max_var = float(np.max(var_per_param))
    min_var = float(np.min(var_per_param))
    return mean_var, max_var, min_var, var_per_param


def run_single_n(
    n: int,
    rng: np.random.Generator,
    n_theta_samples: int,
    t_list: np.ndarray,
    workers: int,
    progress_every: int,
) -> Tuple[VarianceResult, np.ndarray, np.ndarray]:
    restricted_labels, restricted_generators = build_restricted_generators(n)
    full_labels, full_generators = build_full_generators(n)
    all_labels, all_basis = build_all_pauli_basis(n)

    dim = 2**n
    rho0 = random_density_matrix(dim, rng)
    observable = random_observable_from_pauli_basis(all_basis, rng)

    # Synthetic data generated from the restricted ansatz.
    d_restricted = len(restricted_generators)
    theta_true = rng.normal(loc=0.0, scale=1.0 / np.sqrt(d_restricted), size=d_restricted)
    h_true = hamiltonian_from_theta(theta_true, restricted_generators)
    y_data = np.array(
        [np.trace(observable @ quantum_flow(h_true, rho0, t)).real for t in t_list],
        dtype=float,
    )

    # Separate RNG streams for the two models.
    restricted_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
    full_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))

    mean_r, max_r, min_r, var_r = gradient_variance_fixed_rho_o(
        restricted_generators,
        rho0,
        observable,
        y_data,
        t_list,
        restricted_rng,
        n_theta_samples=n_theta_samples,
        workers=workers,
        progress_label=f"n={n} restricted",
        progress_every=progress_every,
    )
    mean_f, max_f, min_f, var_f = gradient_variance_fixed_rho_o(
        full_generators,
        rho0,
        observable,
        y_data,
        t_list,
        full_rng,
        n_theta_samples=n_theta_samples,
        workers=workers,
        progress_label=f"n={n} full",
        progress_every=progress_every,
    )

    ratio_mean = mean_r / mean_f if mean_f != 0 else np.inf
    ratio_max = max_r / max_f if max_f != 0 else np.inf
    ratio_min = min_r / min_f if min_f != 0 else np.inf

    result = VarianceResult(
        n=n,
        dim_restricted=len(restricted_generators),
        dim_full=len(full_generators),
        mean_restricted=mean_r,
        max_restricted=max_r,
        min_restricted=min_r,
        mean_full=mean_f,
        max_full=max_f,
        min_full=min_f,
        ratio_mean=ratio_mean,
        ratio_max=ratio_max,
        ratio_min=ratio_min,
    )
    return result, var_r, var_f


def save_results_csv(path: Path, results: Sequence[VarianceResult]) -> None:
    fieldnames = list(asdict(results[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def plot_summary(path: Path, results: Sequence[VarianceResult]) -> None:
    ns = [r.n for r in results]
    restricted_means = [r.mean_restricted for r in results]
    full_means = [r.mean_full for r in results]
    ratios = [r.ratio_mean for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(ns, restricted_means, marker="o", color="tab:blue", label="Lie-restricted")
    axes[0].plot(ns, full_means, marker="o", color="tab:red", label="Full")
    axes[0].set_xlabel("n")
    axes[0].set_ylabel("Mean gradient variance")
    axes[0].set_title("Mean variance vs system size")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(ns, ratios, marker="o", color="tab:purple")
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("n")
    axes[1].set_ylabel("Restricted / Full")
    axes[1].set_title("Variance ratio")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_report(results: Sequence[VarianceResult]) -> None:
    header = (
        "n  dimR  dimF   meanR        meanF        ratio(mean)   "
        "maxR         maxF         ratio(max)"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.n:<2d} "
            f"{r.dim_restricted:<5d} "
            f"{r.dim_full:<5d} "
            f"{r.mean_restricted:>11.4e} "
            f"{r.mean_full:>11.4e} "
            f"{r.ratio_mean:>12.4f} "
            f"{r.max_restricted:>11.4e} "
            f"{r.max_full:>11.4e} "
            f"{r.ratio_max:>12.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Appendix variance experiment for Lie-restricted vs full models."
    )
    parser.add_argument(
        "--n-values",
        nargs="+",
        type=int,
        default=[3, 4, 5],
        help="System sizes to evaluate.",
    )
    parser.add_argument(
        "--theta-samples",
        type=int,
        default=1000,
        help="Number of theta samples per model and per n.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker threads used for theta sampling.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress every N completed samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master random seed.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("appendix_variance_results.csv"),
        help="CSV output path.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("appendix_variance_summary.png"),
        help="Summary plot output path.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plot generation.",
    )
    args = parser.parse_args()

    t_list = np.linspace(0.1, 1.0, 10)
    master_rng = np.random.default_rng(args.seed)

    print(
        f"Starting appendix variance experiment with n={args.n_values}, "
        f"theta_samples={args.theta_samples}, workers={args.workers}",
        flush=True,
    )

    results: List[VarianceResult] = []
    for n in args.n_values:
        print(f"\nRunning system size n={n}", flush=True)
        case_rng = np.random.default_rng(master_rng.integers(0, 2**32 - 1))
        result, _, _ = run_single_n(
            n=n,
            rng=case_rng,
            n_theta_samples=args.theta_samples,
            t_list=t_list,
            workers=args.workers,
            progress_every=args.progress_every,
        )
        results.append(result)
        print(
            f"Finished n={n}: mean ratio={result.ratio_mean:.4f}, "
            f"max ratio={result.ratio_max:.4f}, min ratio={result.ratio_min:.4f}",
            flush=True,
        )

    print_report(results)
    save_results_csv(args.csv, results)
    if not args.no_plot:
        plot_summary(args.plot, results)

    print(f"\nSaved CSV to {args.csv}", flush=True)
    if not args.no_plot:
        print(f"Saved plot to {args.plot}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
