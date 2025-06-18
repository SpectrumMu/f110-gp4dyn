from pyexpat import model
from re import X
import torch
import numpy as np
import pickle
from gp_model import MultiOutputGP, MultiOutputSparseGP, MultiOutputStochasticVariationalGP
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

MODEL = 'multioutput'  # 'multioutput', 'sparse', or 'stochastic_variational'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load the data
DATADIR = '/home/mu/workspace/roboracer/data/kine_rand_uniform/'
MODELDIR = '/home/mu/workspace/roboracer/src/gp-ws/models/'

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
# Convert to PyTorch tensors
train_states = torch.tensor(train_states, dtype=torch.float32)
train_controls = torch.tensor(train_controls, dtype=torch.float32)
train_dynamics = torch.tensor(train_dynamics, dtype=torch.float32)

# Assume you only have one friction class, drop the first dim
states = train_states[0]    # (N, 2, 4)
controls = train_controls[0]  # (N, 1, 2)
dynamics = train_dynamics[0]  # (N, 1, 4)

# Select a subset of the data, e.g., N=1000
N = 3000
indices = np.random.choice(states.shape[0], size=N, replace=False)
states = states[indices]
controls = controls[indices]
dynamics = dynamics[indices]

xk = states[:, 0, :]     # (N, 4)
uk = controls[:, 0, :]   # (N, 2)
xk1 = states[:, 1, :]    # (N, 4)
yk = dynamics[:, 0, :]  # (N, 4)

X_train = torch.cat([xk, uk], dim=-1)  # (N, 6)
Y_train = yk                           # (N, 4)

# Split into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X_train, Y_train, test_size=0.2, random_state=42
)

# === Train model ===
# gp_model = MultiOutputGP(X_train, Y_train, device=device)
# gp_model.train(X_train, Y_train, training_iter=100)
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
        model.train(X_train, Y_train, training_iter=100)
    else:
        model.train(num_epochs=[100, 100, 100, 100])

# Save each model
for name, model in models.items():
    with open(MODELDIR + f"gp_model_{name}.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to gp_model_{name}.pkl")

# Predict and evaluate for each model
results = {}
for name, model in models.items():
    Y_pred, Y_std = model.predict(X_test)
    results[name] = (Y_pred, Y_std)

# === Evaluation: plot all models in the same figure ===
num_outputs = Y_test.shape[-1]
fig, axes = plt.subplots(2, num_outputs, figsize=(6 * num_outputs, 10))

colors = {'multioutput': 'blue', 'sparse': 'green', 'stochastic_variational': 'red'}
for i in range(num_outputs):
    ax_hist = axes[0, i] if num_outputs > 1 else axes[0]
    ax_scatter = axes[1, i] if num_outputs > 1 else axes[1]
    for name, (Y_pred, Y_std) in results.items():
        y_true = Y_test[:, i].numpy()
        y_pred = Y_pred[:, i].numpy()
        y_uncert = Y_std[:, i].numpy()
        error = np.abs(y_true - y_pred)
        normalized_error = error / y_uncert

        # Histogram (overlayed)
        ax_hist.hist(normalized_error, bins=30, alpha=0.4, label=name, color=colors[name])
        # Scatter
        ax_scatter.scatter(y_uncert, error, alpha=0.4, label=name, color=colors[name])

    ax_hist.set_xlabel("Normalized Error")
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
plt.savefig("/home/mu/workspace/roboracer/src/gp-ws/evaluate_out/all_outputs_norm_hist_and_scatter_compare.png")
plt.show()

# === Save model ===
exit(0)
with open(MODELDIR + "gp_model.pkl", "wb") as f:
    pickle.dump(gp_model, f)

print("Model saved to gp_model.pkl")

# # === Load model ===
# with open("gp_model.pkl", "rb") as f:
#     loaded_model = pickle.load(f)

# print("Model loaded.")

# # === Predict and Evaluate ===
Y_pred, Y_std = gp_model.predict(X_test)

# === Evaluation ===
num_outputs = Y_test.shape[-1]
fig, axes = plt.subplots(2, num_outputs, figsize=(6 * num_outputs, 10))  # Removed sharey='row'

for i in range(num_outputs):
    y_true = Y_test[:, i].numpy()
    y_pred = Y_pred[:, i].numpy()
    y_uncert = Y_std[:, i].numpy()
    error = np.abs(y_true - y_pred)
    normalized_error = error / y_uncert

    # Top row: normalized error histogram
    ax_hist = axes[0, i] if num_outputs > 1 else axes[0]
    ax_hist.hist(normalized_error, bins=30, alpha=0.7)
    ax_hist.set_xlabel("Normalized Error")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title(f"Output {i}: Norm Error Hist")
    ax_hist.grid(True)

    # Bottom row: scatter plot (absolute error vs. uncertainty)
    ax_scatter = axes[1, i] if num_outputs > 1 else axes[1]
    ax_scatter.scatter(y_uncert, error, alpha=0.5)
    ax_scatter.set_xlabel("Predicted Stddev (Uncertainty)")
    ax_scatter.set_ylabel("Absolute Error")
    ax_scatter.set_title(f"Output {i}: Error vs Uncertainty")
    ax_scatter.grid(True)

plt.tight_layout()
plt.savefig("/home/mu/workspace/roboracer/src/gp-ws/evaluate_out/all_outputs_norm_hist_and_scatter.png")
plt.show()


