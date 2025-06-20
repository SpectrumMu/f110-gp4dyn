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

config = None
with open("./config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Extract Runtime Parameters in config
# Some path
DATADIR = config["global"]["data_folder"]
MODELDIR = config["global"]["model_folder"]
EVALDIR = config["global"]["eval_folder"]
# Model info
IF_NORM = config["compare_config"]["if_norm"]
N_SUBSET = int(config["compare_config"]["N_sub"])
SPLIT = float(config["compare_config"]["train_test_split"])
EPOCH = int(config["compare_config"]["epoch"])

# Create directories if they do not exist
if not os.path.exists(MODELDIR):
    os.makedirs(MODELDIR)
if not os.path.exists(EVALDIR):
    os.makedirs(EVALDIR)

# Increase max CG iterations and adjust tolerance
max_cg_iterations(2000)  # Increase the maximum iterations
cg_tolerance(1e-3)       # Relax the tolerance slightly

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device) 

# Load the data

train_data = np.load(DATADIR + 'train_data.npz')
train_states = train_data['train_states']
train_controls = train_data['train_controls']
train_dynamics = train_data['train_dynamics']

print('train_states shape:', train_states.shape)
print('train_controls shape:', train_controls.shape)
print('train_dynamics shape:', train_dynamics.shape)
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
    X_train, Y_train, test_size=SPLIT, random_state=42
)

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
        model.train(X_train, Y_train, training_iter=EPOCH)
    else:
        model.train(num_epochs=[EPOCH, EPOCH, EPOCH, EPOCH])

# Generate a timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

MODELDIR = MODELDIR + "compare/"
if not os.path.exists(MODELDIR):
    os.makedirs(MODELDIR)
    
MODELDIR += f"{timestamp}/"
if not os.path.exists(MODELDIR):
    os.makedirs(MODELDIR)

# Save each model
for name, model in models.items():
    model_filename = f"gp_model_{name}.pkl"
    with open(MODELDIR + model_filename, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_filename}")

# Save scalers
scalers_filename = f"scalers.pkl"
with open(MODELDIR + scalers_filename, "wb") as f:
    pickle.dump({'x_scaler': x_scaler, 'y_scaler': y_scaler}, f)
print(f"Scalers saved to {scalers_filename}")

# Collect data information
data_info = {
    "train_states_shape": train_states.shape,
    "train_controls_shape": train_controls.shape,
    "train_dynamics_shape": train_dynamics.shape,
    "subset_indices": indices.tolist() if 'indices' in locals() else None,
    "X_train_shape": X_train.shape,
    "Y_train_shape": Y_train.shape,
    "X_test_shape": X_test.shape,
    "Y_test_shape": Y_test.shape,
    "normalization": IF_NORM,
    "timestamp": timestamp,
    "v_max": np.max(train_states[:, 0, 1]),
    "v_min": np.min(train_states[:, 0, 1]),
    "model_filename": model_filename,
    "scaler_filename": scalers_filename
}

# Save data info to a JSON file
data_info_filename = f"data_info.json"
with open(MODELDIR + data_info_filename, "w") as f:
    json.dump(data_info, f, indent=4)

print(f"Data info saved to {data_info_filename}")


# Predict and evaluate for each model
results = {}
for name, model in models.items():
    Y_pred, Y_std = model.predict(X_test)
    Y_pred = y_scaler.inverse_transform(Y_pred.numpy())  # Inverse transform Y_pred if normalized
    Y_std = y_scaler.inverse_transform(Y_std.numpy())  # Inverse transform Y_std if normalized
    results[name] = (Y_pred, Y_std)

Y_test = Y_test.numpy()  # Convert to numpy for evaluation
Y_test = y_scaler.inverse_transform(Y_test)  # Inverse transform Y_test if normalized

# === Evaluation: plot all models in the same figure ===
num_outputs = Y_test.shape[-1]
fig, axes = plt.subplots(2, num_outputs, figsize=(6 * num_outputs, 10))

eval_dir_timestamped = EVALDIR + f"compare_{timestamp}/"
if not os.path.exists(eval_dir_timestamped):
    os.makedirs(eval_dir_timestamped)

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
        
    ax_scatter.set_xlim(left=0)
    ax_scatter.set_ylim(bottom=0)

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
plt.savefig(eval_dir_timestamped + "all_outputs_norm_hist_and_scatter_compare.png")
plt.show()


