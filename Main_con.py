
# ----------------------------------------------------------------------
import numpy as np
from numpy import kron
from scipy.linalg import expm
import itertools

np.random.seed(42)


# ----------------------------------------------------------------------
# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

paulis = [I, X, Y, Z]

def dagger(A):
    return A.conj().T

def comm(A, B):
    return A @ B - B @ A


# ----------------------------------------------------------------------
# Restricted Lie algebra generators
G_restricted = [
    kron(X, I),
    kron(Z, Z),
    kron(I, Y)
]

# Full su(4): all Pauli ⊗ Pauli except identity ⊗ identity
G_full = []
for A, B in itertools.product(paulis, paulis):
    if not (np.allclose(A, I) and np.allclose(B, I)):
        G_full.append(kron(A, B))


# ----------------------------------------------------------------------
def quantum_flow(H, rho, t):
    U = expm(-1j * t * H)
    return U @ rho @ dagger(U)

def loss_and_grad(theta, generators, rho0, O, t_list, y_data):
    """
    Compute loss and gradient w.r.t. theta
    """
    H = sum(theta[k] * generators[k] for k in range(len(theta)))
    grad = np.zeros_like(theta, dtype=float)
    loss = 0.0

    for idx, t in enumerate(t_list):
        rho_t = quantum_flow(H, rho0, t)
        y = np.trace(O @ rho_t).real
        diff = y - y_data[idx]
        loss += diff**2

        U = expm(-1j * t * H)
        for k, Xk in enumerate(generators):
            Xk_t = U @ Xk @ dagger(U)
            drho = -1j * t * comm(Xk_t, rho_t)
            dy = np.trace(O @ drho).real
            grad[k] += 2 * diff * dy

    return loss, grad


# ----------------------------------------------------------------------
def random_density_matrix():
    A = np.random.randn(4,4) + 1j*np.random.randn(4,4)
    rho = A @ dagger(A)
    return rho / np.trace(rho)

def random_observable():
    O = np.zeros((4,4), dtype=complex)
    for A, B in itertools.product(paulis, paulis):
        coeff = np.random.uniform(-1, 1)
        O += coeff * kron(A, B)
    return O

# Time grid
t_list = np.linspace(0.1, 1.0, 10)

# Ground truth Hamiltonian (restricted)
theta_true = np.random.randn(len(G_restricted))
H_true = sum(theta_true[k] * G_restricted[k] for k in range(len(theta_true)))


# ----------------------------------------------------------------------
def gradient_variance(generators, N_samples=2000):
    grads = []

    for _ in range(N_samples):
        rho0 = random_density_matrix()
        O = random_observable()

        # synthetic data
        y_data = [
            np.trace(O @ quantum_flow(H_true, rho0, t)).real
            for t in t_list
        ]

        theta = np.random.randn(len(generators))
        _, grad = loss_and_grad(theta, generators, rho0, O, t_list, y_data)
        grads.append(grad)

    grads = np.array(grads)
    return np.var(grads, axis=0).mean()  # average over parameters


# ----------------------------------------------------------------------
var_restricted = gradient_variance(G_restricted)
var_full = gradient_variance(G_full)

print("Gradient variance (restricted):", var_restricted)
print("Gradient variance (full):      ", var_full)
print("Ratio (restricted / full):     ", var_restricted / var_full)


# ----------------------------------------------------------------------
import numpy as np
from numpy import kron
from scipy.linalg import expm
import itertools
import matplotlib.pyplot as plt

np.random.seed(42)  # 固定随机种子保证可复现

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

paulis = [I, X, Y, Z]

def dagger(A):
    return A.conj().T

def comm(A, B):
    return A @ B - B @ A

# Restricted Lie algebra generators (dim=3)
G_restricted = [
    kron(X, I),
    kron(Z, Z),
    kron(I, Y)
]

# Full su(4) generators (dim=15)
G_full = []
for A, B in itertools.product(paulis, paulis):
    if not (np.allclose(A, I) and np.allclose(B, I)):
        G_full.append(kron(A, B))

def quantum_flow(H, rho, t):
    """Compute U(t)ρU†(t) where U(t) = exp(-itH)"""
    U = expm(-1j * t * H)
    return U @ rho @ dagger(U)

def loss_and_grad(theta, generators, rho0, O, t_list, y_data):
    """Compute loss and gradient w.r.t. theta (only theta is variable)"""
    H = sum(theta[k] * generators[k] for k in range(len(theta)))
    grad = np.zeros_like(theta, dtype=float)
    loss = 0.0

    for idx, t in enumerate(t_list):
        rho_t = quantum_flow(H, rho0, t)
        y = np.trace(O @ rho_t).real
        diff = y - y_data[idx]
        loss += diff**2

        # Gradient calculation (core of Theorem 2)
        U = expm(-1j * t * H)
        for k, Xk in enumerate(generators):
            Xk_t = U @ Xk @ dagger(U)  # Lie代数共轭平移
            drho = -1j * t * comm(Xk_t, rho_t)  # 量子流参数导数
            dy = np.trace(O @ drho).real
            grad[k] += 2 * diff * dy

    return loss, grad

def random_density_matrix():
    """Generate random 4x4 density matrix (positive, trace=1)"""
    A = np.random.randn(4,4) + 1j*np.random.randn(4,4)
    rho = A @ dagger(A)
    return rho / np.trace(rho)

def random_observable():
    """Generate random 4x4 Hermitian observable"""
    O = np.zeros((4,4), dtype=complex)
    for A, B in itertools.product(paulis, paulis):
        coeff = np.random.uniform(-1, 1)
        O += coeff * kron(A, B)
    return O  # 天然Hermitian（实系数Pauli张量积）

# --------------------------
# 核心修改：固定rho0和O，仅采样theta
# --------------------------
def gradient_variance_fixed_rho_O(generators, rho0, O, y_data, t_list, N_theta_samples=5000):
    """
    固定rho0和O，仅对theta采样计算梯度方差
    输出：所有参数的梯度方差均值（对应Theorem 2的Var[∂L/∂θ_k]）
    """
    grads = []
    n_params = len(generators)

    # 仅采样theta（符合Haar期望的定义）
    for _ in range(N_theta_samples):
        theta = np.random.randn(n_params)  # 高斯初始化（紧集上的测度）
        _, grad = loss_and_grad(theta, generators, rho0, O, t_list, y_data)
        grads.append(grad)

    grads = np.array(grads)
    var_per_param = np.var(grads, axis=0)  # 每个参数的梯度方差
    mean_var = np.mean(var_per_param)      # 平均梯度方差（论文报告此值）
    return mean_var, var_per_param

# --------------------------
# 实验配置（严格对应理论）
# --------------------------
# 1. 时间网格（有限时间步，无影响）
t_list = np.linspace(0.1, 1.0, 10)

# 2. 固定rho0和O（关键！）
rho0_fixed = random_density_matrix()
O_fixed = random_observable()

# 3. 生成真实数据（来自restricted Hamiltonian，保证物理合理性）
theta_true = np.random.randn(len(G_restricted))
H_true = sum(theta_true[k] * G_restricted[k] for k in range(len(theta_true)))
y_data = [np.trace(O_fixed @ quantum_flow(H_true, rho0_fixed, t)).real for t in t_list]

# 4. 计算梯度方差（仅theta采样）
var_restricted, var_per_param_restricted = gradient_variance_fixed_rho_O(
    G_restricted, rho0_fixed, O_fixed, y_data, t_list, N_theta_samples=5000
)
var_full, var_per_param_full = gradient_variance_fixed_rho_O(
    G_full, rho0_fixed, O_fixed, y_data, t_list, N_theta_samples=5000
)

# --------------------------
# 输出结果（可直接写入论文）
# --------------------------
print("=== 固定ρ₀和O，仅θ采样的梯度方差 ===")
print(f"Lie-restricted (dim=3) 平均梯度方差: {var_restricted:.4f}")
print(f"Full su(4) (dim=15)     平均梯度方差: {var_full:.4f}")
print(f"方差比值 (restricted/full): {var_restricted / var_full:.4f}")

# --------------------------
# 绘制Fig.2（论文用图）
# --------------------------
plt.rcParams['font.size'] = 12
plt.figure(figsize=(8, 5))

# 绘制每个参数的梯度方差
params_restricted = np.arange(1, len(var_per_param_restricted)+1)
params_full = np.arange(1, len(var_per_param_full)+1)

plt.scatter(params_restricted, var_per_param_restricted,
            label='Lie-restricted ($\\mathfrak{g} = \\mathrm{span}\{X\\otimes I, Z\\otimes Z, I\\otimes Y\}$)',
            color='blue', s=30, alpha=0.8)
plt.scatter(params_full, var_per_param_full,
            label='Full $\\mathfrak{su}(4)$',
            color='red', s=15, alpha=0.6)

# 绘制均值线
plt.axhline(y=var_restricted, color='blue', linestyle='--', alpha=0.5, label='Restricted mean')
plt.axhline(y=var_full, color='red', linestyle='--', alpha=0.5, label='Full mean')

plt.xlabel('Parameter Index $k$')
plt.ylabel('$\\mathrm{Var}_\\theta[\\partial \\mathcal{L}/\\partial \\theta_k]$')
plt.title('Gradient Variance per Parameter (2-Qubit Toy System)')
plt.legend(loc='upper right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('fig2_gradient_variance.pdf', dpi=300, bbox_inches='tight')
plt.show()


# ----------------------------------------------------------------------
import numpy as np
from numpy import kron
from scipy.linalg import expm
import itertools
import matplotlib.pyplot as plt

np.random.seed(42)  # 固定随机种子保证可复现

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

paulis = [I, X, Y, Z]

def dagger(A):
    return A.conj().T

def comm(A, B):
    return A @ B - B @ A

# Restricted Lie algebra generators (dim=3)
G_restricted = [
    kron(X, I),
    kron(Z, Z),
    kron(I, Y)
]

# Full su(4) generators (dim=15)
G_full = []
for A, B in itertools.product(paulis, paulis):
    if not (np.allclose(A, I) and np.allclose(B, I)):
        G_full.append(kron(A, B))

def quantum_flow(H, rho, t):
    """Compute U(t)ρU†(t) where U(t) = exp(-itH)"""
    U = expm(-1j * t * H)
    return U @ rho @ dagger(U)

def loss_and_grad(theta, generators, rho0, O, t_list, y_data):
    """Compute loss and gradient w.r.t. theta (only theta is variable)"""
    H = sum(theta[k] * generators[k] for k in range(len(theta)))
    grad = np.zeros_like(theta, dtype=float)
    loss = 0.0

    for idx, t in enumerate(t_list):
        rho_t = quantum_flow(H, rho0, t)
        y = np.trace(O @ rho_t).real
        diff = y - y_data[idx]
        loss += diff**2

        # Gradient calculation (core of Theorem 2)
        U = expm(-1j * t * H)
        for k, Xk in enumerate(generators):
            Xk_t = U @ Xk @ dagger(U)  # Lie代数共轭平移
            drho = -1j * t * comm(Xk_t, rho_t)  # 量子流参数导数
            dy = np.trace(O @ drho).real
            grad[k] += 2 * diff * dy

    return loss, grad

def random_density_matrix():
    """Generate random 4x4 density matrix (positive, trace=1)"""
    A = np.random.randn(4,4) + 1j*np.random.randn(4,4)
    rho = A @ dagger(A)
    return rho / np.trace(rho)

def random_observable():
    """Generate random 4x4 Hermitian observable"""
    O = np.zeros((4,4), dtype=complex)
    for A, B in itertools.product(paulis, paulis):
        coeff = np.random.uniform(-1, 1)
        O += coeff * kron(A, B)
    return O  # 天然Hermitian（实系数Pauli张量积）

# --------------------------
# 核心修改：固定rho0和O，仅采样theta
# --------------------------
def gradient_variance_fixed_rho_O(generators, rho0, O, y_data, t_list, N_theta_samples=5000):
    """
    固定rho0和O，仅对theta采样计算梯度方差
    输出：所有参数的梯度方差均值（对应Theorem 2的Var[∂L/∂θ_k]）
    """
    grads = []
    n_params = len(generators)

    # 仅采样theta（符合Haar期望的定义）
    for _ in range(N_theta_samples):
        theta = np.random.randn(n_params)  # 高斯初始化（紧集上的测度）
        _, grad = loss_and_grad(theta, generators, rho0, O, t_list, y_data)
        grads.append(grad)

    grads = np.array(grads)
    var_per_param = np.var(grads, axis=0)  # 每个参数的梯度方差
    mean_var = np.mean(var_per_param)      # 平均梯度方差（论文报告此值）
    return mean_var, var_per_param

# --------------------------
# 实验配置（严格对应理论）
# --------------------------
# 1. 时间网格（有限时间步，无影响）
t_list = np.linspace(0.1, 1.0, 10)

# 2. 固定rho0和O（关键！）
rho0_fixed = random_density_matrix()
O_fixed = random_observable()

# 3. 生成真实数据（来自restricted Hamiltonian，保证物理合理性）
theta_true = np.random.randn(len(G_restricted))
H_true = sum(theta_true[k] * G_restricted[k] for k in range(len(theta_true)))
y_data = [np.trace(O_fixed @ quantum_flow(H_true, rho0_fixed, t)).real for t in t_list]

# 4. 计算梯度方差（仅theta采样）
var_restricted, var_per_param_restricted = gradient_variance_fixed_rho_O(
    G_restricted, rho0_fixed, O_fixed, y_data, t_list, N_theta_samples=5000
)
var_full, var_per_param_full = gradient_variance_fixed_rho_O(
    G_full, rho0_fixed, O_fixed, y_data, t_list, N_theta_samples=5000
)

# --------------------------
# 输出结果（可直接写入论文）
# --------------------------
print("=== 固定ρ₀和O，仅θ采样的梯度方差 ===")
print(f"Lie-restricted (dim=3) 平均梯度方差: {var_restricted:.4f}")
print(f"Full su(4) (dim=15)     平均梯度方差: {var_full:.4f}")
print(f"方差比值 (restricted/full): {var_restricted / var_full:.4f}")

# --------------------------
# 绘制Fig.2（论文用图）- 修正转义字符警告
# --------------------------
plt.rcParams['font.size'] = 12
plt.rcParams['text.usetex'] = False  # 禁用LaTeX渲染以避免依赖问题
plt.figure(figsize=(8, 5))

# 绘制每个参数的梯度方差
params_restricted = np.arange(1, len(var_per_param_restricted)+1)
params_full = np.arange(1, len(var_per_param_full)+1)

# 核心修正：用原始字符串r""包裹LaTeX标签，避免转义警告
plt.scatter(params_restricted, var_per_param_restricted,
            label=r'Lie-restricted ($\mathfrak{g} = \mathrm{span}\{X\otimes I, Z\otimes Z, I\otimes Y\}$)',
            color='blue', s=30, alpha=0.8)
plt.scatter(params_full, var_per_param_full,
            label=r'Full $\mathfrak{su}(4)$',
            color='red', s=15, alpha=0.6)

# 绘制均值线
plt.axhline(y=var_restricted, color='blue', linestyle='--', alpha=0.5, label=r'Restricted mean')
plt.axhline(y=var_full, color='red', linestyle='--', alpha=0.5, label=r'Full mean')

plt.xlabel(r'Parameter Index $k$')
plt.ylabel(r'$\mathrm{Var}_\theta[\partial \mathcal{L}/\partial \theta_k]$')
plt.title(r'Gradient Variance per Parameter (2-Qubit Toy System)')
plt.legend(loc='upper right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('fig2_gradient_variance.pdf', dpi=300, bbox_inches='tight')
plt.show()


# ----------------------------------------------------------------------
import numpy as np
from numpy import kron
from scipy.linalg import expm
import itertools
import matplotlib.pyplot as plt

# ======================================================
# Global settings
# ======================================================
np.random.seed(42)

# ------------------------------------------------------
# Pauli matrices
# ------------------------------------------------------
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

paulis = [I, X, Y, Z]

# ------------------------------------------------------
# Basic algebra utilities
# ------------------------------------------------------
def dagger(A):
    return A.conj().T

def comm(A, B):
    return A @ B - B @ A

# ======================================================
# Lie generators
# ======================================================
# Restricted generators (basis of parameterized subspace)
G_restricted = [
    kron(X, I),
    kron(Z, Z),
    kron(I, Y)
]

# Full su(4) generators (Pauli tensor products excluding identity)
G_full = []
for A, B in itertools.product(paulis, paulis):
    if not (np.allclose(A, I) and np.allclose(B, I)):
        G_full.append(kron(A, B))

# ======================================================
# Quantum dynamics
# ======================================================
def quantum_flow_from_U(U, rho):
    return U @ rho @ dagger(U)

def loss_and_grad(theta, generators, rho0, O, t_list, y_data):
    """
    Compute loss and gradient ∂L/∂θ_k
    Fully consistent with Theorem (fixed rho0, O; only θ varies)
    """
    d = len(generators)
    H = sum(theta[k] * generators[k] for k in range(d))

    loss = 0.0
    grad = np.zeros(d, dtype=float)

    for idx, t in enumerate(t_list):
        U = expm(-1j * t * H)
        rho_t = quantum_flow_from_U(U, rho0)

        y = np.trace(O @ rho_t).real
        diff = y - y_data[idx]
        loss += diff**2

        for k, Xk in enumerate(generators):
            Xk_t = U @ Xk @ dagger(U)
            drho = -1j * t * comm(Xk_t, rho_t)
            dy = np.trace(O @ drho).real
            grad[k] += 2.0 * diff * dy

    return loss, grad

# ======================================================
# Random objects (properly normalized)
# ======================================================
def random_density_matrix():
    """Hilbert–Schmidt random density matrix"""
    A = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
    rho = A @ dagger(A)
    return rho / np.trace(rho)

def random_observable():
    """Random Hermitian observable, HS-normalized"""
    O = np.zeros((4, 4), dtype=complex)
    for A, B in itertools.product(paulis, paulis):
        coeff = np.random.uniform(-1.0, 1.0)
        O += coeff * kron(A, B)

    # Hilbert–Schmidt normalization (CRITICAL)
    O /= np.linalg.norm(O, 'fro')
    return O

# ======================================================
# Gradient variance experiment
# ======================================================
def gradient_variance_fixed_rho_O(
    generators, rho0, O, y_data, t_list, N_theta_samples=5000
):
    """
    Fixed (rho0, O), sample theta only.
    Returns:
        mean variance over parameters
        per-parameter variances
    """
    d = len(generators)
    grads = []

    # Haar-compatible scaling: Var(theta_k) = 1/d
    sigma = 1.0 / np.sqrt(d)

    for _ in range(N_theta_samples):
        theta = np.random.randn(d) * sigma
        _, grad = loss_and_grad(theta, generators, rho0, O, t_list, y_data)
        grads.append(grad)

    grads = np.asarray(grads)
    var_per_param = np.var(grads, axis=0)
    mean_var = np.mean(var_per_param)

    return mean_var, var_per_param

# ======================================================
# Experiment configuration
# ======================================================
t_list = np.linspace(0.1, 1.0, 10)

rho0 = random_density_matrix()
O = random_observable()

# Generate synthetic data from restricted model
theta_true = np.random.randn(len(G_restricted)) / np.sqrt(len(G_restricted))
H_true = sum(theta_true[k] * G_restricted[k] for k in range(len(theta_true)))
y_data = [
    np.trace(O @ quantum_flow_from_U(expm(-1j * t * H_true), rho0)).real
    for t in t_list
]

# ======================================================
# Run experiments
# ======================================================
var_restricted, var_k_restricted = gradient_variance_fixed_rho_O(
    G_restricted, rho0, O, y_data, t_list
)

var_full, var_k_full = gradient_variance_fixed_rho_O(
    G_full, rho0, O, y_data, t_list
)

# ======================================================
# Output (paper-ready)
# ======================================================
print("=== Fixed ρ₀ and O; θ-sampling only ===")
print(f"Lie-restricted (dim=3): mean Var = {var_restricted:.6f}")
print(f"Full su(4)     (dim=15): mean Var = {var_full:.6f}")
print(f"Ratio (restricted / full): {var_restricted / var_full:.4f}")

# ======================================================
# Plot (Fig. 2) - 核心修复：替换不兼容的LaTeX命令，优化图例/标签
# ======================================================
plt.figure(figsize=(8, 5))

# 绘制受限模型的参数方差
plt.scatter(
    np.arange(1, len(var_k_restricted) + 1),
    var_k_restricted,
    color='blue',
    s=40,
    label=r'Lie-restricted ($\dim \mathfrak{g} = 3$)'  # 修复1：\mathfrak{g}替代\mathcal{g}
)

# 绘制全su(4)模型的参数方差
plt.scatter(
    np.arange(1, len(var_k_full) + 1),
    var_k_full,
    color='red',
    s=25,
    alpha=0.7,
    label=r'Full $\mathfrak{su}(4)$'
)

# 绘制各模型的均值方差水平线
plt.axhline(var_restricted, color='blue', linestyle='--', alpha=0.6, label='Restricted mean')
plt.axhline(var_full, color='red', linestyle='--', alpha=0.6, label='Full su(4) mean')

# 核心修复2：替换含\mathcal的ylabel，用基础LaTeX命令实现相同含义
plt.xlabel(r'Parameter index $k$')
plt.ylabel(r'$\mathrm{Var}_\theta[\partial L / \partial \theta_k]$')  # 移除\mathcal{L}，直接用L
plt.title('Gradient Variance per Parameter (2-Qubit Toy System)')
plt.legend(loc='upper right')  # 优化：指定图例位置，避免遮挡点
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('fig2_gradient_variance.png', dpi=300, bbox_inches='tight')  # 优化：防止标签被裁剪
plt.show()


# ----------------------------------------------------------------------
import numpy as np
from numpy import kron
from scipy.linalg import expm
import itertools
import matplotlib.pyplot as plt

# ======================================================
# Global settings (reproducibility)
# ======================================================
np.random.seed(42)

# ------------------------------------------------------
# Pauli matrices
# ------------------------------------------------------
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

paulis = [I, X, Y, Z]

# ------------------------------------------------------
# Basic algebra utilities
# ------------------------------------------------------
def dagger(A):
    return A.conj().T

def comm(A, B):
    return A @ B - B @ A

# ======================================================
# Lie generators
# ======================================================
# Restricted generators (parameterized subspace;
# Lie algebra is generated by these elements)
G_restricted = [
    kron(X, I),
    kron(Z, Z),
    kron(I, Y)
]

# Full su(4) generators (Pauli tensor products excluding identity)
G_full = []
for A, B in itertools.product(paulis, paulis):
    if not (np.allclose(A, I) and np.allclose(B, I)):
        G_full.append(kron(A, B))

# ======================================================
# Quantum dynamics
# ======================================================
def quantum_flow_from_U(U, rho):
    return U @ rho @ dagger(U)

def loss_and_grad(theta, generators, rho0, O, t_list, y_data):
    """
    Compute loss and gradient ∂L/∂θ_k.

    IMPORTANT:
    - rho0 and O are fixed
    - only θ is randomized
    - consistent with Var_θ definition in the theory
    """
    d = len(generators)
    H = sum(theta[k] * generators[k] for k in range(d))

    loss = 0.0
    grad = np.zeros(d, dtype=float)

    for idx, t in enumerate(t_list):
        U = expm(-1j * t * H)
        rho_t = quantum_flow_from_U(U, rho0)

        y = np.trace(O @ rho_t).real
        diff = y - y_data[idx]
        loss += diff**2

        for k, Xk in enumerate(generators):
            Xk_t = U @ Xk @ dagger(U)
            drho = -1j * t * comm(Xk_t, rho_t)
            dy = np.trace(O @ drho).real
            grad[k] += 2.0 * diff * dy

    return loss, grad

# ======================================================
# Random objects (properly normalized)
# ======================================================
def random_density_matrix():
    """Hilbert–Schmidt random density matrix"""
    A = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
    rho = A @ dagger(A)
    return rho / np.trace(rho)

def random_observable():
    """Random Hermitian observable, HS-normalized"""
    O = np.zeros((4, 4), dtype=complex)
    for A, B in itertools.product(paulis, paulis):
        coeff = np.random.uniform(-1.0, 1.0)
        O += coeff * kron(A, B)

    # Hilbert–Schmidt normalization (CRITICAL for variance comparison)
    O /= np.linalg.norm(O, 'fro')
    return O

# ======================================================
# Gradient variance experiment
# ======================================================
def gradient_variance_fixed_rho_O(
    generators, rho0, O, y_data, t_list, N_theta_samples=5000
):
    """
    Fixed (rho0, O), sample θ only.

    This corresponds exactly to Var_θ[∂L/∂θ_k]
    appearing in the theoretical analysis.
    """
    d = len(generators)
    grads = []

    # Haar-compatible energy normalization:
    # E||H||^2 = O(1) for all d
    sigma = 1.0 / np.sqrt(d)

    for _ in range(N_theta_samples):
        theta = np.random.randn(d) * sigma
        _, grad = loss_and_grad(theta, generators, rho0, O, t_list, y_data)
        grads.append(grad)

    grads = np.asarray(grads)

    # Per-parameter variance
    var_per_param = np.var(grads, axis=0)

    # Summary statistics (for table)
    mean_var = np.mean(var_per_param)
    max_var = np.max(var_per_param)
    min_var = np.min(var_per_param)

    return mean_var, max_var, min_var, var_per_param

# ======================================================
# Experiment configuration (finite-size sanity check)
# ======================================================
t_list = np.linspace(0.1, 1.0, 10)

rho0 = random_density_matrix()
O = random_observable()

# Synthetic data generated from restricted Hamiltonian
theta_true = np.random.randn(len(G_restricted)) / np.sqrt(len(G_restricted))
H_true = sum(theta_true[k] * G_restricted[k] for k in range(len(theta_true)))
y_data = [
    np.trace(O @ quantum_flow_from_U(expm(-1j * t * H_true), rho0)).real
    for t in t_list
]

# ======================================================
# Run experiments
# ======================================================
mean_r, max_r, min_r, var_k_restricted = gradient_variance_fixed_rho_O(
    G_restricted, rho0, O, y_data, t_list
)

mean_f, max_f, min_f, var_k_full = gradient_variance_fixed_rho_O(
    G_full, rho0, O, y_data, t_list
)

# ======================================================
# Output (paper-ready, table-ready)
# ======================================================
print("=== Fixed ρ₀ and O; θ-sampling only (finite-size sanity check) ===")
print(f"Lie-restricted (dim=3):")
print(f"  mean Var = {mean_r:.6e}")
print(f"  max  Var = {max_r:.6e}")
print(f"  min  Var = {min_r:.6e}")

print(f"Full su(4) (dim=15):")
print(f"  mean Var = {mean_f:.6e}")
print(f"  max  Var = {max_f:.6e}")
print(f"  min  Var = {min_f:.6e}")

print(f"Ratio (restricted / full):")
print(f"  mean ratio = {mean_r / mean_f:.4f}")
print(f"  max  ratio = {max_r / max_f:.4f}")
print(f"  min  ratio = {min_r / min_f:.4f}")

# ======================================================
# Plot (Fig. 2)
# ======================================================
plt.figure(figsize=(8, 5))

plt.scatter(
    np.arange(1, len(var_k_restricted) + 1),
    var_k_restricted,
    color='blue',
    s=40,
    label=r'Lie-restricted ($\dim \mathfrak{g}=3$)'
)

plt.scatter(
    np.arange(1, len(var_k_full) + 1),
    var_k_full,
    color='red',
    s=25,
    alpha=0.7,
    label=r'Full $\mathfrak{su}(4)$'
)

plt.axhline(mean_r, color='blue', linestyle='--', alpha=0.6, label='Restricted mean')
plt.axhline(mean_f, color='red', linestyle='--', alpha=0.6, label='Full mean')

plt.xlabel(r'Parameter index $k$')
plt.ylabel(r'$\mathrm{Var}_\theta[\partial L / \partial \theta_k]$')
plt.title('Gradient Variance per Parameter (2-Qubit Toy System)')
plt.legend(loc='upper right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('fig2_gradient_variance.png', dpi=300, bbox_inches='tight')
plt.show()

