import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from linear_operator.settings import max_cg_iterations, cg_tolerance

import json, os, sys, datetime, logging, time
import dotenv
from omegaconf import OmegaConf

from gp_model import MultiOutputExactGP, MultiOutputSparseGP, MultiOutputStochasticVariationalGP, MultiOutputSparseHeteroskedasticGP
from utils.utils import prepare, load_yaml_config
from utils.logger import setup_logger

dotenv.load_dotenv()  # automatically loads from .env in current dir
ws_home = os.getenv("MY_WS_HOME")

def main():
    # === Load configuration ===
    config = OmegaConf.load("./config/config.yaml")

    # Apply command-line overrides
    cli_conf = OmegaConf.from_dotlist(sys.argv[1:])
    config = OmegaConf.merge(config, cli_conf)
    config_train = config.gp_train

    # Extract Runtime Parameters in config
    path_dict = prepare(config, "train")
    global DATADIR, MODELDIR, EVALDIR, LOGDIR, TIME_STAMP
    DATADIR = os.path.join(ws_home, config.global_config.data_folder)
    MODELDIR = path_dict["model_dir"]
    EVALDIR = path_dict["eval_folder"]
    LOGDIR = path_dict["log_folder"]
    TIME_STAMP = path_dict["timestamp"]

    # Model info
    IF_NORM = config_train.if_norm
    MODEL_TYPE = config_train.model.type
    SPLIT = config_train.train_test_split
    EPOCH = config_train.epoch
    LEARNING_RATE = config_train.model.learning_rate
    BATCH_SIZE = config_train.model.get("batch_size", 512)
    PATIENCE = config_train.model.get("patience", 50)

    # Increase max CG iterations and adjust tolerance
    max_cg_iterations(2000)  # Increase the maximum iterations
    cg_tolerance(1e-3)       # Relax the tolerance slightly

    # Set up logging
    logger = setup_logger(
        name="train_logger",
        log_dir=LOGDIR
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    X_train, Y_train, X_test, Y_test, x_scaler, y_scaler = data_load(logger, IF_NORM=IF_NORM, SPLIT=SPLIT)

    # === Create or Load the Model ===
    gp_model = None
    try:
        gp_model = create_model(config.gp_train, X_train, Y_train, device)
        logger.info(f"Created model of type {config.gp_train.model.type} on device {device}")
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        return
    logger.info(f"Training model with {X_train.shape[0]} training samples...")
    
    # === Train the model ===
    start_time = time.time()
    losses = None
    if MODEL_TYPE == 0:
        losses = gp_model.train(X_train, Y_train, training_iter=EPOCH, logger=logger, lr=LEARNING_RATE)
    elif MODEL_TYPE == 1:
        losses = gp_model.train(
            X_train=X_train,
            Y_train=Y_train,
            num_epochs=EPOCH, 
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE, 
            logger=logger)
    elif MODEL_TYPE == 2:
        losses = gp_model.train(
            X_train=X_train,
            Y_train=Y_train,
            num_epochs=EPOCH, 
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE, 
            logger=logger)
    elif MODEL_TYPE == 3:
        losses = gp_model.train(
            X_train=X_train,
            Y_train=Y_train,
            num_epochs=EPOCH, 
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE, 
            logger=logger)
    end_time = time.time()
    training_time = end_time - start_time
    logger.info(f"Training completed in {end_time - start_time:.2f} seconds.")

    # === Save model ===
    name = "multioutput" if MODEL_TYPE == 0 else "sparse" if MODEL_TYPE == 1 else "stochastic_variational" if MODEL_TYPE == 2 else "sparse_heteroskedastic"
    model_filename = f"gp_model_{name}.pkl"
    with open(MODELDIR + model_filename, "wb") as f:
        pickle.dump(gp_model, f)

    # Save scalers
    scaler_filename = "scaler.pkl"
    with open(MODELDIR + scaler_filename, "wb") as f:
        pickle.dump({'x_scaler': x_scaler, 'y_scaler': y_scaler}, f)

    logger.info(f"Model saved to {model_filename}")
    logger.info(f"Scalers saved to {scaler_filename}")

    # Collect data information
    data_info = {
        "train_data_shape": X_train.shape,
        "test_data_shape": X_test.shape,
        "num_train_samples": X_train.shape[0],
        "subset_indices": None,  # Remove undefined indices reference
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
    with open(EVALDIR + data_info_filename, "w") as f:
        json.dump(data_info, f, indent=4)

    logger.info(f"Data info saved to {data_info_filename}")

    # Collect the model configuration and train parameters
    model_config = {
        "model_type": name,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "patience": PATIENCE,
        "epoch": EPOCH,
        "if_norm": IF_NORM,
        "train_test_split": SPLIT,
        "device": str(device),
        "training_time": training_time,
    }
    
    # Save model configuration to a JSON file
    model_config_filename = "model_config.json"
    with open(MODELDIR + model_config_filename, "w") as f:
        json.dump(model_config, f, indent=4)
    with open(EVALDIR + model_config_filename, "w") as f:
        json.dump(model_config, f, indent=4)
        
    logger.info(f"Model configuration saved to {model_config_filename}")

    # # === Predict and Evaluate ===
    Y_pred, Y_std, _, _ = gp_model.predict(X_test)

    if IF_NORM:
        # Restore predictions to original scale
        Y_pred = y_scaler.inverse_transform(Y_pred.numpy())
        Y_std  = y_scaler.inverse_transform(Y_std.numpy())
        Y_test = y_scaler.inverse_transform(Y_test.numpy())
    else:
        Y_pred = Y_pred.numpy()
        Y_std  = Y_std.numpy()
        Y_test = Y_test.numpy()

    # === Evaluation ===
    evaluate_error_uncertainty(Y_test, Y_pred, Y_std)
    logger.info("Training and evaluation completed successfully.")

    if losses is not None:
        plot_losses(losses)
        logger.info("Training loss plot saved.")

    pass

def data_load(logger, IF_NORM=True, SPLIT=0.2):
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
    states = train_states[0]    # (N, 2, 5)config["model"]["type"]
    controls = train_controls[0]  # (N, 1, 2)
    dynamics = train_dynamics[0]  # (N, 1, 5)

    xk = states[:, 0, :]     # (N, 5)
    uk = controls[:, 0, :]   # (N, 2)
    xk1 = states[:, 1, :]    # (N, 5)
    yk = dynamics[:, 0, :]  # (N, 5)

    X_train = np.concatenate([xk, uk], axis=-1)  # (N, 7)
    Y_train = yk  # (N, 5), select columns 0, 1, 3, 4

    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"Y_train shape: {Y_train.shape}")

    # Normalize X_train and Y_train
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

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

    return X_train, Y_train, X_test, Y_test, x_scaler, y_scaler

def create_model(config, X_train, Y_train, device, load_from_file=None):
    """
    Create a GP model based on the specified type.
    
    Args:
        model_type (int): Type of GP model to create.
        X_train (torch.Tensor): Training input data.
        Y_train (torch.Tensor): Training output data.
        device (torch.device): Device to run the model on.
        load_from_file (str, optional): Path to load a pre-trained model from.
        
    Returns:
        gp_model: The created or loaded GP model.
    """
    if load_from_file:
        with open(load_from_file, "rb") as f:
            gp_model = pickle.load(f)
            gp_model.to(device)
            return gp_model

    model_type = int(config.model.type)
    learning_rate = float(config.model.learning_rate)
    inducing = int(config.model.inducing)
    independent = bool(config.model.independent)

    if model_type == 0:
        return MultiOutputExactGP(X_train, Y_train, device=device)
    elif model_type == 1:
        return MultiOutputSparseGP(
            input_dim=X_train.shape[1],
            output_dim=Y_train.shape[1],
            num_latents=Y_train.shape[1],
            independent=independent,
            num_inducing_points=inducing,
            device=device
        )
    elif model_type == 2:
        return MultiOutputStochasticVariationalGP(
            input_dim=X_train.shape[1],
            output_dim=Y_train.shape[1],
            num_latents=Y_train.shape[1],
            independent=independent,
            num_inducing_points=inducing,
            device=device
        )
    elif model_type == 3:
        return MultiOutputSparseHeteroskedasticGP(
            input_dim=X_train.shape[1],
            output_dim=Y_train.shape[1],
            num_latents=Y_train.shape[1],
            num_inducing_points=inducing,
            device=device
        )
    else:
        raise ValueError("Invalid model type specified.")

def evaluate_error_uncertainty(Y_test, Y_pred, Y_std):
    """
    Evaluate the model's predictions against the true values.
    
    Args:
        Y_test (torch.Tensor): True output values.
        Y_pred (torch.Tensor): Predicted output values.
        Y_std (torch.Tensor): Predicted standard deviations (uncertainty).
        
    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    # === Evaluation ===
    num_outputs = Y_test.shape[-1]
    fig, axes = plt.subplots(2, num_outputs, figsize=(6 * num_outputs, 10))  # Removed sharey='row'

    for i in range(num_outputs):
        y_true = Y_test[:, i]
        y_pred = Y_pred[:, i]
        y_uncert = Y_std[:, i]
        error = np.abs(y_true - y_pred)
        # normalized_error = error / y_uncert

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
        
        # ax_scatter.set_xlim(left=0)
        # ax_scatter.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(EVALDIR + "all_outputs_norm_hist_and_scatter.png")
    # plt.show()
    
    pass

def plot_losses(losses):
    """
    Plot training losses over epochs.
    
    Args:
        losses (list or np.ndarray): List of loss values per epoch.
        save_path (str): Path to save the loss plot.
    """
    losses = np.array(losses)
    losses = np.exp(losses)  # Convert log losses back to normal scale if needed
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.yscale('log')  # Log scale for better visibility
    plt.grid(True)
    plt.legend()
    plt.savefig(EVALDIR + "training_loss.png")
    plt.close()


if __name__ == "__main__":
    main()
