# SGD-Operator-Learning

This repository contains code for operator learning using operator-valued kernel stochastic gradient descent (SGD). The learning task involves approximating solution operators arising from the two-dimensional incompressible Navier–Stokes equations in the vorticity–stream function formulation.

## Problem Description

We consider the mapping from the external forcing function \( g \) to the vorticity field \( u(10, \cdot) \), where the governing equation is:


$$
\begin{aligned}
\frac{\partial u}{\partial t}+(\nu\cdot\nabla)u-\nu\Delta u &= g, \\
u &= -\Delta\phi,\quad \int_D\phi=0,\\
\nu &= \left(\frac{\partial\phi}{\partial x_{2}},-\frac{\partial\phi}{\partial x_{1}}\right),
\end{aligned}
$$



with domain \( D = [0,2\pi]^2 \) and viscosity \( \nu = 0.025 \). The forcing \( g \) is sampled from a Gaussian process \( \mathcal{GP}(0,(-\Delta + 3^2 I)^{-4}) \).

## Dataset

We use the dataset provided by [de2022cost](https://arxiv.org/abs/2202.05862), which contains 40,000 i.i.d. input–output pairs obtained by numerically solving the Navier–Stokes equations on a \(64 \times 64\) grid. The dataset is publicly available at:

📎 **CaltechDATA**: [https://data.caltech.edu/records/fp3ds-kej20](https://data.caltech.edu/records/fp3ds-kej20)

We split the data into training, validation, and test sets with a 70%–15%–15% ratio. PCA is applied to reduce both input and output dimensions to 128 components before training.

## Method

We use stochastic gradient descent (SGD) in a Reproducing Kernel Hilbert Space (RKHS) defined by a Matérn kernel multiplied by the identity operator. Both fixed and decaying step sizes are explored.

## Code Structure

- `main.py`: Main training script
- `data_loader.py`: Data loading and preprocessing
- `kernels.py`: Kernel definitions
- `sgd_solver.py`: Implementation of kernel SGD

## Citation

If you find this repository useful, please cite our work.
