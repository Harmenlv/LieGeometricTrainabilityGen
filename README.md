Lie-Geometric Trainability of Quantum Dynamical Systems
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()
Official implementation for the research paper:
📄 Lie-Geometric Trainability of Quantum Dynamical Systems: Avoiding Barren Plateaus via Low-Dimensional Lie Subalgebras
Physics Scripta (Accepted Manuscript, 2026)

DOI: https://doi.org/10.1088/1402-4896/ae7d6d

Journal Page: https://iopscience.iop.org/article/10.1088/1402-4896/ae7d6d

GitHub Repository: https://github.com/Harmenlv/LieGeometricTrainability.git

---
📌 Overview
This repository contains all source code and numerical experiments to verify the theoretical framework proposed in the paper.
💡 Core Statement
Lie closure determines the trainability of quantum dynamical systems.
The Lie algebra constructed by Hamiltonian operators dominates the dimension of physically reachable state orbits. This property further controls gradient distribution, and ultimately decides whether barren plateaus emerge in quantum learning tasks. Restricting dynamics to low-dimensional Lie subalgebras is an effective approach to avoid exponentially vanishing gradients.

---
🔍 Framework & Figure Explanation
Two key figures are placed in the repository root directory to illustrate the core logic and model structure:


1. framework.png : Core Theoretical Chain
The complete causal logic of this work is summarized as follows:
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
Model Trainability
- Generator selection defines the Lie algebra of the quantum system;
- The dimension of Lie closure limits the maximum size of reachable state orbits under unitary evolution;
- Orbit dimension governs the statistical properties of gradient moments;
- Gradient variance is the critical metric for judging barren plateaus;
- Sufficient gradient variance guarantees stable trainability of quantum models.
2. Generator_Structure.png : Algebra Generator Design
This figure shows two types of generator sets adopted in comparative experiments:
✅ Lie-Restricted Generators
Composed of local single-qubit operators and two-qubit coupling operators, forming a polynomial-dimensional Lie subalgebra. Its dimension scales polynomially with the number of qubits.
✅ Full $$\mathfrak{su}(2^n)$$ Generators
Consists of all Pauli tensor products (excluding global identity operator), forming the complete Lie algebra for $$n$$-qubit systems. Its dimension scales exponentially with the number of qubits.
All comparative experiments are built on these two generator families.

---
📊 Main Theoretical Results
Regime
Gradient Variance
Lie-restricted ($$\dim(\mathfrak g_n)=\mathrm{poly}(n)$$)
$$\mathrm{Var}(\partial\mathcal L/\partial\theta_k)\ge C/\mathrm{poly}(n)$$
Fully expressive ($$\mathfrak{su}(2^n)$$)
$$\mathrm{Var}(\partial\mathcal L/\partial\theta_k)=O(2^{-n})$$
The variance enhancement factor satisfies
$$R_n= \Omega\!\left( \frac{2^n}{\dim(\mathfrak g_n)} \right),$$
demonstrating an exponential trainability advantage of Lie-restricted quantum models.

---
📁 Repository Structure
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
│   ├── appendix_variance_experiment.py
│   ├── appendix_variance_experiment4.py
│   ├── appendix_variance_experiment5.py
│   ├── collective_spin_variance_experiment.py
│   └── initial_residual_scaling_experiment.py
│
├── data/
│   ├── lie_variance_results.csv
│   ├── full_variance_results.csv
│   ├── collective_spin_variance_results.csv
│   └── initial_residual_scaling_results.csv
│
├── figures/
│   ├── collective_spin_variance_summary.png
│   └── initial_residual_scaling_summary.png
│
└── docs/
    └── paper.pdf

---
🧪 Experiments Description
All experiments correspond to the theoretical conclusions in the accepted paper.
1. Two-Qubit Sanity Check
File:experiments/appendix_variance_experiment.py
Compare gradient variance between the restricted Lie subalgebra and full $$\mathfrak{su}(4)$$ algebra on a minimal 2-qubit system.
2. Gradient Variance Scaling
File: experiments/collective_spin_variance_experiment.py
Verify the scaling laws:
$$\mathrm{Var}_{Lie} = \Omega\left(1/\mathrm{poly}(n)\right), \quad \mathrm{Var}_{Full} = O(2^{-n})$$
Output:
- data/collective_spin_variance_results.csv
- figures/collective_spin_variance_summary.png
3. Initial Residual Scaling
File: experiments/initial_residual_scaling_experiment.py
Validate the fundamental assumption in the paper (Assumption IV.10):
$$\mathbb E_\theta \big[|f(\theta)-y|^2\big] \ge \beta >0$$
The training residual is uniformly bounded away from zero, independent of system size.
Output:
- data/initial_residual_scaling_results.csv
- figures/initial_residual_scaling_summary.png
4. Variance Ratio Analysis
Files:
- experiments/appendix_variance_experiment4.py
- experiments/appendix_variance_experiment5.py
Calculate the empirical variance ratio:
$$R_n = \mathrm{Var}_{Lie}/\mathrm{Var}_{Full}$$
Compare numerical results with the theoretical prediction $$R_n \sim \dfrac{2^n}{\dim(\mathfrak{g}_n)}$$.

---
⚙️ Installation & Environment
Step 1: Clone the repository
git clone https://github.com/Harmenlv/LieGeometricTrainability.git
cd LieGeometricTrainability
Step 2: Install dependencies
pip install -r requirements.txt
📋 Dependencies List (requirements.txt)
numpy
scipy
matplotlib
pandas
tqdm

---
▶️ Reproduce Experimental Results
Run the main script to execute all experiments automatically:
python Main.py
Alternative entry script:
python Main_con.py
✅ Expected Outputs
- CSV files: Raw numerical results (stored in data/ folder)
- PNG figures: Visualization plots (stored in figures/ folder)
- Gradient variance scaling plots
- Training residual scaling plots
- Variance ratio comparison plots

---
📝 Citation
If you use this code or refer to this work in your research, please cite the following paper:
@article{Shao2026LieTrainability,
  title={Lie-Geometric Trainability of Quantum Dynamical Systems: Avoiding Barren Plateaus via Low-Dimensional Lie Subalgebras},
  author={Shao, Haijian and Wu, Yujie and Deng, Xing and Jiang, Yingtao},
  journal={Physics Scripta},
  year={2026},
  doi={10.1088/1402-4896/ae7d6d},
  url={https://iopscience.iop.org/article/10.1088/1402-4896/ae7d6d}
}

---
📜 License
This project is licensed under the MIT License. See the LICENSE file for full license text.

---
👨‍🔬 Academic Homepage
For publications, citations, and research updates, please visit my Google Scholar profile:
🔗 https://scholar.google.com/citations?user=d3mvChQAAAAJ&hl=en
If you find this project useful in your research, I would greatly appreciate it if you cite my related publications.
Thank you for your support and citations!
Have a nice day!
