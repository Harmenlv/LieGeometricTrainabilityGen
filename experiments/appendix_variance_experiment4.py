#!/usr/bin/env python3
"""
针对李限制模型（Lie-restricted）的多样本重复实验。
从 random_numbers.txt 读取 100 个种子，并将每次实验结果保存至 CSV。
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

# 定义 Pauli 矩阵
PAULI_MATS: Dict[str, np.ndarray] = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


@dataclass
class LieVarianceResult:
    n: int
    seed: int
    dim_restricted: int
    mean_restricted: float
    max_restricted: float
    min_restricted: float
    elapsed_time: float


def dagger(a: np.ndarray) -> np.ndarray:
    return a.conj().T


def comm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b - b @ a


def kron_all(ops: Sequence[np.ndarray]) -> np.ndarray:
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def build_restricted_generators(n: int) -> Tuple[List[str], List[np.ndarray]]:
    labels, mats = [], []
    for i in range(n):
        ops = [PAULI_MATS["I"]] * n
        ops[i] = PAULI_MATS["X"]
        labels.append(f"X{i + 1}")
        mats.append(kron_all(ops))
    for i in range(n - 1):
        ops = [PAULI_MATS["I"]] * n
        ops[i] = PAULI_MATS["Z"]
        ops[i + 1] = PAULI_MATS["Z"]
        labels.append(f"Z{i + 1}Z{i + 2}")
        mats.append(kron_all(ops))
    return labels, mats


def random_density_matrix(dim: int, rng: np.random.Generator) -> np.ndarray:
    a = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    rho = a @ dagger(a)
    rho /= np.trace(rho)
    return rho


def random_observable_from_generators(generators: Sequence[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    coeffs = rng.uniform(-1.0, 1.0, size=len(generators))
    observable = sum(c * m for c, m in zip(coeffs, generators))
    observable = 0.5 * (observable + dagger(observable))
    fro_norm = np.linalg.norm(observable, ord="fro")
    return observable / fro_norm if fro_norm != 0 else observable


def quantum_flow(h: np.ndarray, rho: np.ndarray, t: float) -> np.ndarray:
    u = expm(-1j * t * h)
    return u @ rho @ dagger(u)


def get_restricted_gradient(theta: np.ndarray, generators: Sequence[np.ndarray], rho0: np.ndarray,
                            observable: np.ndarray, t_list: Sequence[float], y_data: np.ndarray) -> np.ndarray:
    h = sum(c * g for c, g in zip(theta, generators))
    grad = np.zeros(len(generators), dtype=float)
    for idx, t in enumerate(t_list):
        u = expm(-1j * t * h)
        rho_t = u @ rho0 @ dagger(u)
        diff = np.trace(observable @ rho_t).real - y_data[idx]
        for k, xk in enumerate(generators):
            drho = -1j * t * comm(u @ xk @ dagger(u), rho_t)
            grad[k] += 2.0 * diff * np.trace(observable @ drho).real
    return grad


def run_single_experiment(n: int, seed: int, n_theta_samples: int, t_list: np.ndarray) -> LieVarianceResult:
    start_time = time.perf_counter()
    rng = np.random.default_rng(seed)
    _, generators = build_restricted_generators(n)
    dim = 2 ** n
    d_r = len(generators)

    rho0 = random_density_matrix(dim, rng)
    observable = random_observable_from_generators(generators, rng)

    theta_true = rng.normal(0.0, 1.0 / np.sqrt(d_r), size=d_r)
    h_true = sum(c * g for c, g in zip(theta_true, generators))
    y_data = np.array([np.trace(observable @ quantum_flow(h_true, rho0, t)).real for t in t_list])

    grads = np.empty((n_theta_samples, d_r), dtype=float)
    sigma = 1.0 / np.sqrt(d_r)
    for s in range(n_theta_samples):
        theta = rng.normal(0.0, sigma, size=d_r)
        grads[s] = get_restricted_gradient(theta, generators, rho0, observable, t_list, y_data)

    var_per_param = np.var(grads, axis=0)
    return LieVarianceResult(
        n=n, seed=seed, dim_restricted=d_r,
        mean_restricted=float(np.mean(var_per_param)),
        max_restricted=float(np.max(var_per_param)),
        min_restricted=float(np.min(var_per_param)),
        elapsed_time=time.perf_counter() - start_time
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-values", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--theta-samples", type=int, default=500)
    parser.add_argument("--input-seeds", type=str, default="random_numbers.txt")
    parser.add_argument("--output-csv", type=str, default="lie_variance_results.csv")
    args = parser.parse_args()

    # 读取种子文件
    if not Path(args.input_seeds).exists():
        print(f"错误: 找不到文件 {args.input_seeds}，请先生成种子文件。")
        return

    with open(args.input_seeds, "r") as f:
        seeds = [int(line.strip()) for line in f if line.strip()]

    t_list = np.linspace(0.1, 1.0, 5)
    field_names = [f.name for f in fields(LieVarianceResult)]

    print(f"开始重复实验：读取 {len(seeds)} 个种子，针对 n={args.n_values} 进行测试")
    print(f"结果将保存至: {args.output_csv}")
    print("-" * 60)

    with open(args.output_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()

        for n in args.n_values:
            print(f"\n正在处理 n={n}:")
            for i, seed in enumerate(seeds):
                res = run_single_experiment(n, seed, args.theta_samples, t_list)
                writer.writerow(asdict(res))
                csvfile.flush()  # 实时写入硬盘防止丢失
                if (i + 1) % 10 == 0:
                    print(f"  已完成 {i + 1}/{len(seeds)} 次重复实验 (Seed: {seed})...")

    print(f"\n实验全部完成。数据已存入 {args.output_csv}")


if __name__ == "__main__":
    main()