import numpy as np
import itertools
import matplotlib.pyplot as plt
from scipy.linalg import expm
from tqdm import tqdm
from functools import reduce

# Reproducibility
np.random.seed(0)

# ======================
# Pauli Matrices & Utils
# ======================
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
pauli_dict = {"I": I, "X": X, "Y": Y, "Z": Z}

def dagger(A):
    return A.conj().T

def comm(A, B):
    return A @ B - B @ A

def kron_n(ops):
    return reduce(np.kron, ops)

def pauli_tensor(string):
    ops = [pauli_dict[c] for c in string]
    return kron_n(ops)

# ======================
# Generator Families
# ======================
def restricted_generators(n):
    gens = []
    # Local X
    for i in range(n):
        s = ["I"] * n
        s[i] = "X"
        gens.append(pauli_tensor(s))
    # Nearest-neighbor ZZ coupling
    for i in range(n - 1):
        s = ["I"] * n
        s[i] = "Z"
        s[i + 1] = "Z"
        gens.append(pauli_tensor(s))
    return gens

def random_pauli_subset(n, size):
    labels = ["I", "X", "Y", "Z"]
    gens = []
    while len(gens) < size:
        s = "".join(np.random.choice(labels) for _ in range(n))
        if s != "I" * n:
            gens.append(pauli_tensor(s))
    return gens

# ======================
# Random Quantum Objects
# ======================
def random_density(dim):
    A = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    rho = A @ dagger(A)
    return rho / np.trace(rho)

def random_observable(dim):
    A = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    O = A + dagger(A)
    return O / np.linalg.norm(O, "fro")

# ======================
# Gradient & Variance
# ======================
def gradient(theta, generators, rho0, O, t_list, y_data):
    d = len(generators)
    H = sum(theta[k] * generators[k] for k in range(d))
    grad = np.zeros(d)
    for j, t in enumerate(t_list):
        U = expm(-1j * t * H)
        rho = U @ rho0 @ dagger(U)
        y = np.trace(O @ rho).real
        diff = y - y_data[j]
        for k, Xk in enumerate(generators):
            Xkt = U @ Xk @ dagger(U)
            drho = -1j * t * comm(Xkt, rho)
            dy = np.trace(O @ drho).real
            grad[k] += 2 * diff * dy
    return grad

def estimate_variance(generators, rho0, O, t_list, y_data, n_samples=300):
    d = len(generators)
    sigma = 1.0 / np.sqrt(d)
    grads = []
    for _ in range(n_samples):
        theta = np.random.randn(d) * sigma
        grads.append(gradient(theta, generators, rho0, O, t_list, y_data))
    grads = np.array(grads)
    return np.mean(np.var(grads, axis=0))

# ======================
# Experiments
# ======================
def toy_2qubit_check():
    """2-qubit sanity check"""
    n = 2
    dim = 2 ** n
    G_res = restricted_generators(n)
    G_full = random_pauli_subset(n, 15)
    rho = random_density(dim)
    O = random_observable(dim)
    t_list = np.linspace(0.1, 1.0, 5)

    theta_true = np.random.randn(len(G_res)) / np.sqrt(len(G_res))
    H_true = sum(theta_true[k] * G_res[k] for k in range(len(G_res)))
    y_data = [
        np.trace(O @ (expm(-1j * t * H_true) @ rho @ dagger(expm(-1j * t * H_true)))).real
        for t in t_list
    ]

    v_res = estimate_variance(G_res, rho, O, t_list, y_data)
    v_full = estimate_variance(G_full, rho, O, t_list, y_data)
    print(f"[Toy 2-qubit] Restricted Var: {v_res:.6f}")
    print(f"[Toy 2-qubit] Full Var:      {v_full:.6f}\n")

def scaling_experiment(n_list):
    """Gradient variance scaling with qubit number"""
    var_res_list = []
    var_full_list = []
    ratio_list = []
    t_list = np.linspace(0.1, 1.0, 5)

    for n in tqdm(n_list, desc="Scaling Experiment"):
        dim = 2 ** n
        G_res = restricted_generators(n)
        G_full = random_pauli_subset(n, min(4 ** n - 1, 60))
        rho = random_density(dim)
        O = random_observable(dim)

        theta_true = np.random.randn(len(G_res)) / np.sqrt(len(G_res))
        H_true = sum(theta_true[k] * G_res[k] for k in range(len(G_res)))
        y_data = [
            np.trace(O @ (expm(-1j * t * H_true) @ rho @ dagger(expm(-1j * t * H_true)))).real
            for t in t_list
        ]

        vr = estimate_variance(G_res, rho, O, t_list, y_data)
        vf = estimate_variance(G_full, rho, O, t_list, y_data)
        var_res_list.append(vr)
        var_full_list.append(vf)
        ratio_list.append(vr / vf)

        print(f"n={n:2d} | Restricted: {vr:.4e} | Full: {vf:.4e} | Ratio: {vr/vf:.3f}")

    # Plot scaling
    plt.figure(figsize=(7, 5))
    plt.plot(n_list, var_res_list, "o-", label="Lie restricted")
    plt.plot(n_list, var_full_list, "s-", label="Full approx")
    plt.yscale("log")
    plt.xlabel("Qubits")
    plt.ylabel("Gradient variance (log scale)")
    plt.title("Gradient Variance Scaling")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("scaling_variance.png", dpi=300)
    plt.show()

    # Plot ratio
    plt.figure(figsize=(7, 5))
    plt.plot(n_list, ratio_list, "o-")
    plt.xlabel("Qubits")
    plt.ylabel("Variance Ratio (restricted / full)")
    plt.title("Variance Ratio")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("variance_ratio.png", dpi=300)
    plt.show()

    return var_res_list, var_full_list, ratio_list

def projection_energy_experiment(n_list):
    """Lie projection energy"""
    energies = []
    for n in tqdm(n_list, desc="Projection Energy"):
        dim = 2 ** n
        G = restricted_generators(n)
        rho = random_density(dim)
        O = random_observable(dim)
        e_list = []
        for _ in range(200):
            theta = np.random.randn(len(G))
            H = sum(theta[k] * G[k] for k in range(len(G)))
            U = expm(-1j * H)
            Orot = dagger(U) @ O @ U
            C = comm(rho, Orot)
            proj = 0.0
            for Xk in G:
                proj += np.abs(np.trace(dagger(Xk) @ C)) ** 2
            e_list.append(proj)
        energies.append(np.mean(e_list))

    plt.figure(figsize=(7, 5))
    plt.plot(n_list, energies, "o-")
    plt.xlabel("Qubits")
    plt.ylabel("Projection Energy")
    plt.title("Lie Projection Energy")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("projection_energy.png", dpi=300)
    plt.show()
    return energies

def dimension_ablation(n=5):
    """Ablation on generator dimension"""
    dim = 2 ** n
    rho = random_density(dim)
    O = random_observable(dim)
    t_list = np.linspace(0.1, 1.0, 5)
    dim_candidates = [5, 10, 20, 40, 80]
    var_list = []

    for d in tqdm(dim_candidates, desc="Dimension Ablation"):
        G = random_pauli_subset(n, d)
        theta_true = np.random.randn(d) / np.sqrt(d)
        H_true = sum(theta_true[k] * G[k] for k in range(d))
        y_data = [
            np.trace(O @ (expm(-1j * t * H_true) @ rho @ dagger(expm(-1j * t * H_true)))).real
            for t in t_list
        ]
        var_list.append(estimate_variance(G, rho, O, t_list, y_data))

    plt.figure(figsize=(7, 5))
    plt.plot(dim_candidates, var_list, "o-")
    plt.yscale("log")
    plt.xlabel("Generator Dimension")
    plt.ylabel("Gradient variance (log scale)")
    plt.title("Generator Dimension Ablation")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("dimension_ablation.png", dpi=300)
    plt.show()

    for d, v in zip(dim_candidates, var_list):
        print(f"Gen Dim {d:3d} | Variance: {v:.4e}")
    return dim_candidates, var_list

# ======================
# Main Entry
# ======================
if __name__ == "__main__":
    # 1. 2-qubit sanity check
    toy_2qubit_check()

    # 2. Scaling experiment
    qubit_range = [2, 3, 4, 5, 6]
    scaling_experiment(qubit_range)

    # 3. Projection energy
    projection_energy_experiment(qubit_range)

    # 4. Generator dimension ablation
    dimension_ablation(n=5)

    print("\nAll experiments completed.")