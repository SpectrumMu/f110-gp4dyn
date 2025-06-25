from ast import mod
from math import e
from re import X
import torch
import numpy as np
import pickle
from gp_model import MultiOutputGP, MultiOutputSparseGP
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# from sklearn.metrics import mean_squared_error  # noqa: E402
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils.utils import prepare, load_yaml_config
from utils.logger import setup_logger
import yaml
import json
import datetime
import os
import logging

def main():
    # === Load configuration ===
    config = load_yaml_config("./config/config.yaml")
    
    global DATADIR, MODELDIR, EVALDIR, LOGDIR, TIME_STAMP
    # MODELDIR = config["global"]["model_folder"]
    EVALDIR = config["global"]["eval_folder"]
    DATADIR = config["global"]["data_folder"]
    LOGDIR = config["global"]["log_folder"]
    LOGDIR = os.path.join(LOGDIR, "eval_logs/")
    if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)
    
    date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    TIME_STAMP = date_time
    EVALDIR = os.path.join(EVALDIR, f"eval_{date_time}/")
    if not os.path.exists(EVALDIR):
        os.makedirs(EVALDIR)
        
    # SKIP LOGGING
    
    IF_NORM = config["gp_eval"]["if_norm"]
    MODEL_TYPE = int(config["gp_eval"]["model_type"])
    # SPLIT = float(config["gp_eval"]["train_test_split"])
    EVAL_TYPE = int(config["gp_eval"]["eval_type"])
    name = "multioutput" if MODEL_TYPE == 0 else "sparse" if MODEL_TYPE == 1 else "stochastic_variational"
    model_name = "gp_model" + ("_" + name if name else "")
    model_name += ".pkl"
    scaler_name = "scaler.pkl"

    MODELDIR = config["gp_eval"]["model_dir"]
    
    logger = setup_logger(
        name="eval_logger",
        log_dir=LOGDIR
    )
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # # === Load model ===
    loaded_model = None
    x_scaler = None
    y_scaler = None
    with open(MODELDIR + model_name, "rb") as f:
        loaded_model = pickle.load(f)
    with open(MODELDIR + scaler_name, "rb") as f:
        scalers = pickle.load(f)
        x_scaler = scalers["x_scaler"]
        y_scaler = scalers["y_scaler"]
    # loaded_model.to(device)

    logger.info("Model loaded.")

    # === Load data ===
    X_train, X_test, Y_train, Y_test = data_load(x_scaler, y_scaler, IF_NORM, logger, device)

    if EVAL_TYPE == 0:
        eval_type_0(loaded_model, X_test, Y_test, x_scaler, y_scaler)
    else:
        eval_type_1(loaded_model, X_train, Y_train, X_test, logger)
    pass

def data_load(x_scaler, y_scaler, IF_NORM, logger, device):
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
    # Convert to PyTorch tensors

    # Assume you only have one friction class, drop the first dim
    states = train_states[0]    # (N, 2, 4)
    controls = train_controls[0]  # (N, 1, 2)
    dynamics = train_dynamics[0]  # (N, 1, 4)

    xk = states[:, 0, :]     # (N, 4)
    uk = controls[:, 0, :]   # (N, 2)
    xk1 = states[:, 1, :]    # (N, 4)
    yk = dynamics[:, 0, :]  # (N, 4)

    X_train = np.concatenate([xk, uk], axis=-1)  # (N, 6)
    Y_train = yk                           # (N, 4)
    
    if IF_NORM:
        X_train = x_scaler.transform(X_train)
        Y_train = y_scaler.transform(Y_train)

    # Split into train and test sets
    _, X_test, _, Y_test = train_test_split(
        X_train, Y_train, test_size=0.5, random_state=42
    )
    
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    Y_train = torch.tensor(Y_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    Y_test = torch.tensor(Y_test, dtype=torch.float32).to(device)
    
    return X_train, X_test, Y_train, Y_test

def eval_type_0(loaded_model, X_test, Y_test, x_scaler=None, y_scaler=None):
    # # === Predict and Evaluate ===
    Y_pred, Y_std, _, _ = loaded_model.predict(X_test)

    Y_pred = Y_pred.cpu().numpy()
    Y_std = Y_std.cpu().numpy()
    
    Y_test = Y_test.cpu().numpy()
    
    Y_pred = y_scaler.inverse_transform(Y_pred)
    Y_std = y_scaler.inverse_transform(Y_std)
    Y_test = y_scaler.inverse_transform(Y_test)
    
    # === Evaluation ===
    num_outputs = Y_test.shape[-1]
    fig, axes = plt.subplots(2, num_outputs, figsize=(6 * num_outputs, 10))  # Removed sharey='row'

    for i in range(num_outputs):
        y_true = Y_test[:, i]
        y_pred = Y_pred[:, i]
        y_uncert = Y_std[:, i]
        error = np.abs(y_true - y_pred)

        # Top row: normalized error histogram
        ax_hist = axes[0, i] if num_outputs > 1 else axes[0]
        ax_hist.hist(error, bins=30, alpha=0.7)
        ax_hist.set_xlabel("Error")
        ax_hist.set_ylabel("Count")
        ax_hist.set_title(f"Output {i}: Error Hist")
        ax_hist.grid(True)

        # Bottom row: scatter plot (absolute error vs. uncertainty)
        ax_scatter = axes[1, i] if num_outputs > 1 else axes[1]
        ax_scatter.scatter(y_uncert, error, alpha=0.5)
        ax_scatter.set_xlabel("Predicted Stddev (Uncertainty)")
        ax_scatter.set_ylabel("Absolute Error")
        ax_scatter.set_title(f"Output {i}: Error vs Uncertainty")
        ax_scatter.grid(True)

    plt.tight_layout()
    plt.savefig(f"{EVALDIR}/all_outputs_norm_hist_and_scatter.png")
    # plt.show()
    
    pass

def eval_type_1(loaded_model, X_train, Y_train, X_test, logger):
    X_train_np = X_train.cpu().numpy()
    Y_train_np = Y_train.cpu().numpy()
    X_test_np = X_test.cpu().numpy()

    idx = np.random.randint(0, X_train_np.shape[0])
    X_train_mean = X_train_np[idx]
    
    for i in range(X_train_np.shape[1]):
        X_train_fake = np.ones_like(X_train_np) * X_train_mean
        X_train_fake[:, i] = X_train_np[:, i]

        X_test_fake = np.ones((1000, X_train_np.shape[1])) * X_train_mean
        X_test_fake[:, i] = np.linspace(
            np.min(X_test_np[:, i]), np.max(X_test_np[:, i]), X_test_fake.shape[0]
        )
        
        # X_train_fake_tensor = torch.tensor(X_train_fake, dtype=torch.float32).to(loaded_model.device)
        X_test_fake_tensor = torch.tensor(X_test_fake, dtype=torch.float32).to(loaded_model.device)

        # Make predictions
        Y_mean, Y_std, Y_lower, Y_upper = loaded_model.predict(X_test_fake_tensor)
        Y_mean = Y_mean.cpu().numpy()
        Y_std = Y_std.cpu().numpy()
        Y_lower = Y_lower.cpu().numpy()
        Y_upper = Y_upper.cpu().numpy()
        
        plt.figure(figsize=(12, 8))
        plt.suptitle(f"Feature Ablation Study for Dim {i}", fontsize=16)

        for j in range(Y_mean.shape[1]):
            plt.subplot(2, 2, j + 1)
            plt.fill_between(
                X_test_fake[:, i],
                Y_lower[:, j],
                Y_upper[:, j],
                alpha=0.3,
                label='Confidence Interval'
            )
            plt.plot(X_test_fake[:, i], Y_mean[:, j], 'r-', label='Prediction')
            # plt.scatter(X_train_np[:, i], Y_train_np[:, j], s=10, label='Training Data')
            plt.xlabel('Test Sample Index')
            plt.ylabel(f'Output {j}')
            plt.title(f'Output {j} Prediction with Feature {i} Varied')
            plt.grid()
            plt.ylim(-3, 3)
            plt.legend()
        
        logger.info(f"Feature ablation for dimension {i} completed and plot saved.")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"{EVALDIR}/feature_ablation_dim_{i}.png")
        # plt.show()
    pass


if __name__ == "__main__":
    main()