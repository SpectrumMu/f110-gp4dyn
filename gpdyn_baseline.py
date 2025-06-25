from sympy import N
import torch
import numpy as np
import pickle
from gp_model import MultiOutputGP, MultiOutputSparseGP, MultiOutputStochasticVariationalGP
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from linear_operator.settings import max_cg_iterations, cg_tolerance
import yaml
import datetime
import os, json
from utils.utils import prepare, load_yaml_config
from utils.logger import setup_logger

def main():
    # === Load configuration ===
    config = load_yaml_config("./config/config.yaml")

    # Extract Runtime Parameters in config
    path_dict = prepare(config, "train", 1)
    # Some path
    global DATADIR, MODELDIR, EVALDIR, LOGDIR, TIME_STAMP
    DATADIR = config["global"]["data_folder"]
    MODELDIR = path_dict["model_dir"]
    EVALDIR = path_dict["eval_folder"]
    LOGDIR = path_dict["log_folder"]
    TIME_STAMP = path_dict["timestamp"]
    # Model info
    IF_NORM = config["compare_config"]["if_norm"]
    # MODEL_TYPE = int(config["compare_config"]["model_type"])
    SPLIT = float(config["compare_config"]["train_test_split"])
    EPOCH = int(config["compare_config"]["epoch"])
    LEARNING_RATE = float(config["compare_config"]["learning_rate"])
    N_SUBSET = None
    N_SUBSET = config["compare_config"].get("N_sub", None)

    # Increase max CG iterations and adjust tolerance
    max_cg_iterations(2000)  # Increase the maximum iterations
    cg_tolerance(1e-3)       # Relax the tolerance slightly

    # Set up logging
    logger = setup_logger(
        name="compare_logger",
        log_dir=LOGDIR
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    X_train, Y_train, X_test, Y_test, x_scaler, y_scaler, indices = data_load(logger, IF_NORM=IF_NORM, SPLIT=SPLIT, N_SUBSET=N_SUBSET)

    # === Train model ===
    # Train all three models on the same data
    models = {
        'multioutput': MultiOutputGP(X_train, Y_train, device=device),
        'sparse': MultiOutputSparseGP(X_train, Y_train, num_inducing=128, device=device),
        'stochastic_variational': MultiOutputStochasticVariationalGP(X_train, Y_train, num_inducing=128, device=device)
    }

    # Train each model
    for name, model in models.items():
        print(f"Training {name} model...")
        if name == 'multioutput':
            model.train(X_train, Y_train, training_iter=EPOCH, logger=logger, lr=LEARNING_RATE)
        else:
            model.train(num_epochs=[EPOCH]*4, lr=LEARNING_RATE, logger=logger)

    # Save each model
    for name, model in models.items():
        model_filename = f"gp_model_{name}.pkl"
        with open(MODELDIR + model_filename, "wb") as f:
            pickle.dump(model, f)
        print(f"Model saved to {model_filename}")

    # Save scalers
    scaler_filename = f"scaler.pkl"
    with open(MODELDIR + scaler_filename, "wb") as f:
        pickle.dump({'x_scaler': x_scaler, 'y_scaler': y_scaler}, f)
    print(f"Scalers saved to {scaler_filename}")

    # Collect data information
    data_info = {
        "train_data_shape": X_train.shape,
        "test_data_shape": X_test.shape,
        "num_train_samples": X_train.shape[0],
        "subset_indices": indices.tolist() if 'indices' in locals() else None,
        "X_train_shape": X_train.shape,
        "Y_train_shape": Y_train.shape,
        "X_test_shape": X_test.shape,
        "Y_test_shape": Y_test.shape,
        "normalization": IF_NORM,
        "timestamp": TIME_STAMP,
        "model_type": name,
        "model_filename": model_filename,
        "scaler_filename": scaler_filename
    }

    # Save data info to a JSON file
    data_info_filename = "data_info.json"
    with open(MODELDIR + data_info_filename, "w") as f:
        json.dump(data_info, f, indent=4)

    logger.info(f"Data info saved to {data_info_filename}")
    
    # === Evaluate model ===
    # Predict and evaluate for each model
    results = {}
    for name, model in models.items():
        Y_pred, Y_std, _, _ = model.predict(X_test)
        Y_pred = y_scaler.inverse_transform(Y_pred.numpy())  # Inverse transform Y_pred if normalized
        Y_std = y_scaler.inverse_transform(Y_std.numpy())  # Inverse transform Y_std if normalized
        results[name] = (Y_pred, Y_std)

    Y_test = Y_test.numpy()  # Convert to numpy for evaluation
    Y_test = y_scaler.inverse_transform(Y_test)  # Inverse transform Y_test if normalized

    evaluate_error_uncertainty(Y_test, results)
    logger.info("Training and evaluation completed successfully.")

    pass

def data_load(logger, IF_NORM=True, SPLIT=0.2, N_SUBSET=3000):
    """
    Loads, validates, preprocesses, normalizes, and splits training data for model training.
    Parameters:
        logger (logging.Logger): Logger object for logging information and errors.
        IF_NORM (bool, optional): If True, normalize the input and output data using StandardScaler. Default is True.
        SPLIT (float, optional): Fraction of data to use for training (between 0 and 1). The remainder is used for testing. Default is 0.2.
    Returns:
        X_train (torch.Tensor): Training input data tensor of shape (num_train_samples, 6).
        Y_train (torch.Tensor): Training output data tensor of shape (num_train_samples, 4).
        X_test (torch.Tensor): Testing input data tensor of shape (num_test_samples, 6).
        Y_test (torch.Tensor): Testing output data tensor of shape (num_test_samples, 4).
        x_scaler (StandardScaler): Fitted scaler object for input data.
        y_scaler (StandardScaler): Fitted scaler object for output data.
    Raises:
        ValueError: If NaN or Inf values are found in the loaded data.
    Notes:
        - Assumes the data is stored in 'train_data.npz' under DATADIR.
        - Assumes only one friction class is present and drops the first dimension accordingly.
        - Concatenates state and control vectors for input features.
        - Splits the data into training and testing sets using sklearn's train_test_split.
    """
    
    train_data = np.load(DATADIR + 'train_data.npz')
    train_states = train_data['train_states']
    train_controls = train_data['train_controls']
    train_dynamics = train_data['train_dynamics']

    logger.info(f"train_states shape: {train_states.shape}")
    logger.info(f"train_controls shape: {train_controls.shape}")
    logger.info(f"train_dynamics shape: {train_dynamics.shape}")
    # train_states: (N, 2, 4)
    # train_controls: (N, 1, 2)
    # train_dynamics: (N, 1, 4)
    # Check for NaN or Inf values
    if np.isnan(train_states).any() or np.isinf(train_states).any():
        raise ValueError("NaN or Inf values found in train_states")
    if np.isnan(train_controls).any() or np.isinf(train_controls).any():
        raise ValueError("NaN or Inf values found in train_controls")
    if np.isnan(train_dynamics).any() or np.isinf(train_dynamics).any():
        raise ValueError("NaN or Inf values found in train_dynamics")

    # Assume you only have one friction class, drop the first dim
    states = train_states[0]    # (N, 2, 4)
    controls = train_controls[0]  # (N, 1, 2)
    dynamics = train_dynamics[0]  # (N, 1, 4)

    # Select a subset of the data, e.g., N=1000
    indices = None
    if N_SUBSET is not None and N_SUBSET < states.shape[0]:
        indices = np.random.choice(states.shape[0], size=N_SUBSET, replace=False)
        states = states[indices]
        controls = controls[indices]
        dynamics = dynamics[indices]

    xk = states[:, 0, :]     # (N, 4)
    uk = controls[:, 0, :]   # (N, 2)
    xk1 = states[:, 1, :]    # (N, 4)
    yk = dynamics[:, 0, :]  # (N, 4)

    X_train = np.concatenate([xk, uk], axis=-1)  # (N, 6)
    Y_train = yk                           # (N, 4)

    # Normalize X_train and Y_train
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    if IF_NORM:
        X_train = x_scaler.fit_transform(X_train)
        Y_train = y_scaler.fit_transform(Y_train)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)

    # Split into train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_train, Y_train, test_size=1-SPLIT, random_state=42
    )

    return X_train, Y_train, X_test, Y_test, x_scaler, y_scaler, indices

def evaluate_error_uncertainty(Y_test, results):
    """
    Evaluate and plot the relationship between prediction error and uncertainty for multiple models.
    
    Parameters:
        Y_test (np.ndarray): True output values of shape (num_samples, num_outputs).
        results (dict): Dictionary where keys are model names and values are tuples of 
                        (Y_pred, Y_std) where Y_pred is the predicted mean and Y_std is the predicted standard deviation.
    
    Returns:
        None
    """
    num_outputs = Y_test.shape[-1]
    fig, axes = plt.subplots(2, num_outputs, figsize=(6 * num_outputs, 10))

    # # === Evaluation: compute error metrics ===
    # for name, (Y_pred, Y_std) in results.items():
    #     error = np.abs(Y_test - Y_pred)
    #     # Compute additional metrics as needed

    colors = {'multioutput': 'blue', 'sparse': 'green', 'stochastic_variational': 'red'}
    for i in range(num_outputs):
        ax_hist = axes[0, i] if num_outputs > 1 else axes[0]
        ax_scatter = axes[1, i] if num_outputs > 1 else axes[1]
        for name, (Y_pred, Y_std) in results.items():
            y_true = Y_test[:, i]
            y_pred = Y_pred[:, i]
            y_uncert = Y_std[:, i]
            
            error = np.abs(y_true - y_pred)
            # normalized_error = error / y_uncert

            # Histogram (overlayed)
            # ax_hist.hist(normalized_error, bins=30, alpha=0.4, label=name, color=colors[name])
            # Plot absolute error histogram (not normalized)
            ax_hist.hist(error, bins=30, alpha=0.4, label=name, color=colors[name])
            # Scatter
            ax_scatter.scatter(y_uncert, error, alpha=0.4, label=name, color=colors[name])
            
        # ax_scatter.set_xlim(left=0)
        # ax_scatter.set_ylim(bottom=0)

        ax_hist.set_xlabel("Error")
        ax_hist.set_ylabel("Count")
        ax_hist.set_title(f"Output {i}: Norm Error Hist")
        ax_hist.grid(True)
        ax_hist.legend()

        ax_scatter.set_xlabel("Predicted Stddev (Uncertainty)")
        ax_scatter.set_ylabel("Absolute Error")
        ax_scatter.set_title(f"Output {i}: Error vs Uncertainty")
        ax_scatter.grid(True)
        ax_scatter.legend()

    plt.tight_layout()
    plt.savefig(EVALDIR + "all_outputs_norm_hist_and_scatter_compare.png")
    plt.show()

if __name__ == "__main__":
    main()

