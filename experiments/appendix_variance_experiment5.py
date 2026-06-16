#!/usr/bin/env python3
"""
针对全参数模型（Full Parameterization）的多样本重复实验。
从 random_numbers.txt 读取 100 个种子，并将每次实验结果保存至 CSV。
严格保持原有计算协议：使用 expm 计算演化，采样 500 个样本。
"""

from __future__ import annotations

import argparse
import itertools
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
class FullVarianceResult:
    n: int
    seed: int
    dim_full: int
    mean_full: float
    max_full: float
    min_full: float
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


def build_full_generators(n: int) -> Tuple[List[str], List[np.ndarray]]:
    """构建 SU(2^n) 的完整 Pauli 展开基（剔除全 I 项）"""
    labels: List[str] = []
    mats: List[np.ndarray] = []
    for ops_labels in itertools.product(["I", "X", "Y", "Z"], repeat=n):
        if all(label == "I" for label in ops_labels):
            continue
        labels.append("".join(ops_labels))
        ops_mats = [PAULI_MATS[l] for l in ops_labels]
        mats.append(kron_all(ops_mats))
    return labels, mats


def random_density_matrix(dim: int, rng: np.random.Generator) -> np.ndarray:
    a = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    rho = a @ dagger(a)
    rho /= np.trace(rho)
    return rho


def random_observable_from_generators(
        generators: Sequence[np.ndarray], rng: np.random.Generator
) -> np.ndarray:
    coeffs = rng.uniform(-1.0, 1.0, size=len(generators))
    observable = np.zeros_like(generators[0], dtype=complex)
    for coeff, mat in zip(coeffs, generators):
        observable = observable + coeff * mat
    observable = 0.5 * (observable + dagger(observable))
    fro_norm = np.linalg.norm(observable, ord="fro")
    return observable / fro_norm if fro_norm != 0 else observable


def quantum_flow(h: np.ndarray, rho: np.ndarray, t: float) -> np.ndarray:
    u = expm(-1j * t * h)
    return u @ rho @ dagger(u)


def get_full_gradient(
        theta: np.ndarray,
        generators: Sequence[np.ndarray],
        rho0: np.ndarray,
        observable: np.ndarray,
        t_list: Sequence[float],
        y_data: np.ndarray,
) -> np.ndarray:
    dim_f = len(generators)
    h = np.zeros_like(generators[0], dtype=complex)
    for coeff, gen in zip(theta, generators):
        h += coeff * gen

    grad = np.zeros(dim_f, dtype=float)
    for idx, t in enumerate(t_list):
        u = expm(-1j * t * h)
        rho_t = u @ rho0 @ dagger(u)
        diff = np.trace(observable @ rho_t).real - y_data[idx]

        for k, xk in enumerate(generators):
            xk_t = u @ xk @ dagger(u)
            drho = -1j * t * comm(xk_t, rho_t)
            dy = np.trace(observable @ drho).real
            grad[k] += 2.0 * diff * dy
    return grad


def run_single_full_experiment(
        n: int,
        seed: int,
        n_theta_samples: int,
        t_list: np.ndarray
) -> FullVarianceResult:
    start_time = time.perf_counter()
    rng = np.random.default_rng(seed)
    _, generators = build_full_generators(n)
    dim = 2 ** n
    d_f = len(generators)

    rho0 = random_density_matrix(dim, rng)
    observable = random_observable_from_generators(generators, rng)

    # 生成合成目标数据
    theta_true = rng.normal(loc=0.0, scale=1.0 / np.sqrt(d_f), size=d_f)
    h_true = np.zeros((dim, dim), dtype=complex)
    for c, g in zip(theta_true, generators): h_true += c * g

    y_data = np.array([np.trace(observable @ quantum_flow(h_true, rho0, t)).real for t in t_list])

    # 采样计算梯度方差
    grads = np.empty((n_theta_samples, d_f), dtype=float)
    sigma = 1.0 / np.sqrt(d_f)

    for s in range(n_theta_samples):
        theta = rng.normal(0.0, sigma, size=d_f)
        grads[s] = get_full_gradient(theta, generators, rho0, observable, t_list, y_data)

    var_per_param = np.var(grads, axis=0)
    return FullVarianceResult(
        n=n, seed=seed, dim_full=d_f,
        mean_full=float(np.mean(var_per_param)),
        max_full=float(np.max(var_per_param)),
        min_full=float(np.min(var_per_param)),
        elapsed_time=time.perf_counter() - start_time
    )


def main():
    parser = argparse.ArgumentParser()
    # 全参数模型建议只跑到 n=5，因为 n=6 的计算量极其巨大
    parser.add_argument("--n-values", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--theta-samples", type=int, default=500)
    parser.add_argument("--input-seeds", type=str, default="random_numbers.txt")
    parser.add_argument("--output-csv", type=str, default="full_variance_results.csv")
    args = parser.parse_args()

    if not Path(args.input_seeds).exists():
        print(f"错误: 找不到文件 {args.input_seeds}")
        return

    with open(args.input_seeds, "r") as f:
        seeds = [int(line.strip()) for line in f if line.strip()]

    t_list = np.linspace(0.1, 1.0, 5)
    field_names = [f.name for f in fields(FullVarianceResult)]

    print(f"开始全参数模型重复实验：读取 {len(seeds)} 个种子")
    print(f"结果保存至: {args.output_csv}")
    print("-" * 60)

    with open(args.output_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()

        for n in args.n_values:
            print(f"\n正在处理 n={n} (参数量: {4 ** n - 1}):")
            for i, seed in enumerate(seeds):
                try:
                    res = run_single_full_experiment(n, seed, args.theta_samples, t_list)
                    writer.writerow(asdict(res))
                    csvfile.flush()
                    if (i + 1) % 10 == 0:
                        print(f"  已完成 {i + 1}/{len(seeds)} 次重复实验 (Seed: {seed})...")
                except MemoryError:
                    print(f"n={n} 内存不足，跳过后续实验。")
                    return

    print(f"\n全参数实验完成。数据已存入 {args.output_csv}")


if __name__ == "__main__":
    main()