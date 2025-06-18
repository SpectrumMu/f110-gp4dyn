import torch
import numpy as np
import pickle
from gp_model import MultiOutputGP, MultiOutputSparseGP, MultiOutputStochasticVariationalGP
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

MODEL = 'sparse'  # 'multioutput', 'sparse', or 'stochastic_variational'
IF_NORM = True  # Whether to normalize the data

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
    X_train, Y_train, test_size=0.2, random_state=42
)

# === Train model ===
gp_model = None
if MODEL == 'multioutput':
    gp_model = MultiOutputGP(X_train, Y_train, device=device)
elif MODEL == 'sparse':
    gp_model = MultiOutputSparseGP(X_train, Y_train, num_inducing=128, device=device)
elif MODEL == 'stochastic_variational':
    gp_model = MultiOutputStochasticVariationalGP(X_train, Y_train, num_inducing=128, device=device)
# gp_model = MultiOutputSparseGP(X_train, Y_train, num_inducing=128)
gp_model.train(num_epochs=[1000,1000,1000,100])

# === Save model ===
with open(MODELDIR + "gp_model.pkl", "wb") as f:
    pickle.dump(gp_model, f)

with open(MODELDIR + "scalers.pkl", "wb") as f:
    pickle.dump({'x_scaler': x_scaler, 'y_scaler': y_scaler}, f)

print("Model saved to gp_model.pkl and scalers saved to scalers.pkl")

# # === Load model ===
# with open("gp_model.pkl", "rb") as f:
#     loaded_model = pickle.load(f)

# with open(MODELDIR + "scalers.pkl", "rb") as f:
#     scalers = pickle.load(f)
# x_scaler = scalers['x_scaler']
# y_scaler = scalers['y_scaler']

# print("Model loaded.")

# # === Predict and Evaluate ===
Y_pred, Y_std = gp_model.predict(X_test)

if IF_NORM:
    # Restore predictions to original scale
    Y_pred_raw = y_scaler.inverse_transform(Y_pred.numpy())
    Y_std_raw = y_scaler.inverse_transform(Y_std.numpy())

    # Example: If you need to restore test data for comparison
    Y_test_raw = y_scaler.inverse_transform(Y_test.numpy())
else:
    Y_pred_raw = Y_pred.numpy()
    Y_std_raw = Y_std.numpy()
    Y_test_raw = Y_test.numpy()


# === Evaluation ===
num_outputs = Y_test.shape[-1]
fig, axes = plt.subplots(2, num_outputs, figsize=(6 * num_outputs, 10))  # Removed sharey='row'

for i in range(num_outputs):
    y_true = Y_test_raw[:, i]
    y_pred = Y_pred_raw[:, i]
    y_uncert = Y_std_raw[:, i]
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

# # Evaluation: MSE
# mse_list = []
# mae_list = []
# r2_list = []

# for i in range(Y_test.shape[-1]):
#     y_true = Y_test[:, i].numpy()
#     y_pred = Y_pred[:, i].numpy()
#     mse = mean_squared_error(y_true, y_pred)
#     mae = mean_absolute_error(y_true, y_pred)
#     r2 = r2_score(y_true, y_pred)
#     mse_list.append(mse)
#     mae_list.append(mae)
#     r2_list.append(r2)
#     print(f"Output {i}: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

# # Aggregate metrics
# print("\nAggregate metrics:")
# print(f"Mean MSE: {np.mean(mse_list):.4f} ± {np.std(mse_list):.4f}")
# print(f"Mean MAE: {np.mean(mae_list):.4f} ± {np.std(mae_list):.4f}")
# print(f"Mean R2:  {np.mean(r2_list):.4f} ± {np.std(r2_list):.4f}")

# # Residual analysis and scatter plots
# for i in range(Y_test.shape[-1]):
#     y_true = Y_test[:, i].numpy()
#     y_pred = Y_pred[:, i].numpy()
#     residuals = y_true - y_pred

#     plt.figure(figsize=(16, 4))
#     plt.subplot(1, 3, 1)
#     plt.plot(y_true, label='True')
#     plt.plot(y_pred, label='Predicted')
#     plt.title(f"Output {i}: Prediction vs True")
#     plt.legend()
#     plt.grid(True)

#     plt.subplot(1, 3, 2)
#     plt.scatter(y_true, y_pred, alpha=0.5)
#     plt.xlabel("True")
#     plt.ylabel("Predicted")
#     plt.title(f"Output {i}: Predicted vs True")
#     plt.grid(True)

#     plt.subplot(1, 3, 3)
#     plt.hist(residuals, bins=30, alpha=0.7)
#     plt.title(f"Output {i}: Residuals")
#     plt.xlabel("Residual")
#     plt.ylabel("Count")
#     plt.grid(True)

#     plt.tight_layout()
#     plt.savefig(f"/home/mu/workspace/src/gp-ws/evaluate_out/output_dim_{i}_eval.png")
#     plt.show()

