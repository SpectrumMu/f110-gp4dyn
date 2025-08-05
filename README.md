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

## Installation and Usage

1. To install the required dependencies, run:
    ```bash
    pip install -r requirements.txt
    ```

1. Configure the dot env file with your environment. First, copy the example file:
    ```bash
    cp .env.example .env
    ```
    Modify the `.env` file with your specific settings
    ```
    MY_WS_HOME=/path/to/your/workspace
    ```

1. Configure the parameters in `config.yaml` according to your needs. This file contains settings for the GP model, data paths, and training parameters.

1. To run the GP regression, execute:
    ```bash
    python gpdyn_train.py
    ```
    Ensure that the data is available in the specified path in `config.yaml`. The script will load the data, preprocess it, train the GP model, and save the results.



## Model Formulation

**IGNORE** the `ref` folder, updated doc at shared Overleaf.

The GP regression problem is defined as:

```math
\mathbf{y} = f(\mathbf{z}) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2) \\ 
f(\mathbf{z}) \sim \mathcal{GP}(m(\mathbf{z}), k(\mathbf{z}, \mathbf{z}'))
```

where the mean function $m(\mathbf{z})$ is typically zero, and the kernel function $k(\mathbf{z}, \mathbf{z}')$ defines the covariance structure of the GP.

```math
\mathbf{z} = \begin{bmatrix}
\delta & v & \dot{\psi} & \beta & a & \dot{\delta}
\end{bmatrix}^\top = \begin{bmatrix} \mathbf{x} \\ \mathbf{u} \end{bmatrix} \in \mathbb{R}^6 \\
\mathbf{y} = \begin{bmatrix}
\dot{\delta} & \dot{v} & \ddot{\psi} & \dot{\beta}
\end{bmatrix}^\top \in \mathbb{R}^4
```

where:
- $\mathbf{z}$: Input features — steering angle ($\delta$), speed ($v$), yaw rate ($\dot{\psi}$), sideslip angle ($\beta$), and control inputs longitudinal acceleration ($a$) and steering speed ($\dot{\delta}$).
- $\mathbf{y}$: Output targets — change rate of steering angle ($\dot{\delta}$), change rate of speed ($\dot{v}$), change rate of yaw rate ($\ddot{\psi}$), and change rate of sideslip angle ($\dot{\beta}$).

The dynamics are computed by 

```math
\mathbf{y} = \frac{\mathbf{x}_{n+1} - \mathbf{x}_{n}}{\Delta t}
```

where $\Delta t$ is the time step between consecutive states.

**Residual Support**: The model supports training with residuals, which only have to replace the dataset.

## Technical Details

### Models 

The code support multiple GP models, including:
- **Exact GP**: For small datasets, providing full covariance.
- **Sparse Variational GP**: For larger datasets, using inducing points to approximate the full GP.
- **Stochastic Variational GP**: Not used, just for reference.



### Folder Structure

```
data/
├── train_data.npz           # Training data file in NumPy format
src/
├── gp-ws/                   # Main GP workspace directory (THIS REPO)
│   ├── gpdyn_train.py       # Main training script for GP dynamics model
│   ├── gp_model.py          # GP model definitions and training utilities
│   ├── config/              # Configuration files for model parameters and paths
│   │   ├── config.yaml      
│   ├── models/              # saved models with datatime and model type
│   ├── evaluate_out/        # directory for evaluation outputs
│   ├── logs/                # directory for log files
│   ├── utils/               # Directory for utility functions
│   │   ├── logger.py        # Logger setup and utilities
│   │   ├── utils.py         # Data loading and preprocessing utilities
│   ├── gp_test.ipynb        # GP functionality test script
│   ├── gp_result.ipynb      # GP result visualization script, visualizing results in ./evaluate_out/
│   ├── training_results.csv # CSV file for training results
│   ├── .env                 # dotenv file
```

The files not mentioned above are currently not used.



### Miscs

- **Data Handling**: The code uses `scikit-learn` for data preprocessing, including normalization and splitting into training and test sets.
    - The `MinMaxScaler` is used to scale the input features and output targets into the range [0, 1], so make sure to adjust the data to avoid NaN values.
- **Logging**: The code uses `tqdm` for progress bars and `logging` for detailed logs of the training process.
- **Configuration**: The `config.yaml` file allows easy configuration of model parameters, data paths, and training settings.
- **Environment Variables**:
    - The `.env` file is used to set environment variables, such as the workspace home directory, which is used to locate data and save results.
- **Visualization**: The code includes visualization of the GP predictions, showing the posterior mean and uncertainty correlations.