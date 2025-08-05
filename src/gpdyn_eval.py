import torch
import numpy as np
import pickle
from gp_model import MultiOutputGP, MultiOutputSparseGP
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# from sklearn.metrics import mean_squared_error  # noqa: E402
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

# # === Load model ===
with open(MODELDIR + "gp_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

print("Model loaded.")

# # === Predict and Evaluate ===
Y_pred = loaded_model.predict(X_test)

# # Evaluation: MSE
# mse = mean_squared_error(Y_test.numpy(), Y_pred.numpy())
# print(f"Test MSE: {mse:.4f}")
# # Optional: Visualize prediction vs truth

# for i in range(Y_test.shape[-1]):
#     plt.figure(figsize=(12, 6))
#     plt.plot(Y_test[:, i].numpy(), label='True')
#     plt.plot(Y_pred[:, i].numpy(), label='Predicted')
#     plt.title(f"Output Dimension {i}")
#     plt.legend()
#     plt.grid(True)
#     plt.savefig("/home/mu/workspace/src/sparse_gp_test/evaluate_out/" + f"output_dim_{i}_prediction.png")
#     plt.show()

# Per-dimension metrics
mse_list = []
mae_list = []
r2_list = []

for i in range(Y_test.shape[-1]):
    y_true = Y_test[:, i].numpy()
    y_pred = Y_pred[:, i].numpy()
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mse_list.append(mse)
    mae_list.append(mae)
    r2_list.append(r2)
    print(f"Output {i}: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

# Aggregate metrics
print("\nAggregate metrics:")
print(f"Mean MSE: {np.mean(mse_list):.4f} ± {np.std(mse_list):.4f}")
print(f"Mean MAE: {np.mean(mae_list):.4f} ± {np.std(mae_list):.4f}")
print(f"Mean R2:  {np.mean(r2_list):.4f} ± {np.std(r2_list):.4f}")

# Residual analysis and scatter plots
for i in range(Y_test.shape[-1]):
    y_true = Y_test[:, i].numpy()
    y_pred = Y_pred[:, i].numpy()
    residuals = y_true - y_pred

    plt.figure(figsize=(16, 4))
    plt.subplot(1, 3, 1)
    plt.plot(y_true, label='True')
    plt.plot(y_pred, label='Predicted')
    plt.title(f"Output {i}: Prediction vs True")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"Output {i}: Predicted vs True")
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.hist(residuals, bins=30, alpha=0.7)
    plt.title(f"Output {i}: Residuals")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"/home/mu/workspace/roboracer/src/gp-ws/evaluate_out/output_dim_{i}_eval.png")
    plt.show()
