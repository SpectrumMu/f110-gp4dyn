# f110-gp4dyn

TODOs:

- [x] Add evaluation like the example in Notebook (in `gp_test.py`)
- [ ] Using different kernels that allow correlation between outputs
- [x] Standardized logging
- [x] Structured folder and config 


This project provides a multi-output Gaussian Process (GP) regression framework for learning and predicting vehicle dynamics, specifically tailored for F1TENTH autonomous racing. Built with GPyTorch, it supports various GP model types, including exact, sparse, and stochastic variational GPs. The pipeline features:

- **Data Loading & Preprocessing:** Handles data validation, normalization, and splitting into training and test sets.
- **Model Training:** Offers multiple GP architectures for multi-output regression with configurable parameters.
- **Evaluation:** Performs predictions on test data, restores original scaling, and visualizes errors and uncertainties.
- **Artifact Saving:** Stores trained models, scalers, and data information for reproducibility.
- **Logging:** Provides detailed logs of training progress and data statistics.

The codebase is modular and configurable, making it suitable for research and experimentation in data-driven vehicle modeling and control.

## Model Formulation

The GP regression problem is defined as:

```math
\mathbf{y} = f(\mathbf{z}) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2) \\
f(\mathbf{z}) \sim \mathcal{GP}(m(\mathbf{z}), k(\mathbf{z}, \mathbf{z}'))
```

where the mean function $m(\mathbf{z})$ is typically zero, and the kernel function $k(\mathbf{z}, \mathbf{z}')$ defines the covariance structure of the GP.

```math
\mathbf{z} = \begin{bmatrix}
\delta & v & \dot{\delta} & \beta & u_1 & u_2
\end{bmatrix}^\top = \begin{bmatrix} \mathbf{x} \\ \mathbf{u} \end{bmatrix} \in \mathbb{R}^6 \\
\mathbf{y} = \begin{bmatrix}
\dot{\delta} & \dot{v} & \dot{\theta} & \dot{\beta}
\end{bmatrix}^\top \in \mathbb{R}^4
```

where:
- $\mathbf{z}$: Input features — steering angle ($\delta$), speed ($v$), rate of change of steering angle ($\dot{\delta}$), sideslip angle ($\beta$), and control inputs ($u_1$, $u_2$).
- $\mathbf{y}$: Output targets — longitudinal acceleration ($a$), rate of change of longitudinal acceleration ($\dot{a}$), yaw angle ($\theta$), and rate of change of yaw angle ($\dot{\theta}$).

The dynamics are computed by 

```math
\mathbf{y} = \frac{\mathbf{x}_{n+1} - \mathbf{x}_{n}}{\Delta t}
```

where $\Delta t$ is the time step between consecutive states.