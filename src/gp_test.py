import numpy as np
from sympy import true
import torch
import matplotlib.pyplot as plt
from gp_model import MultiOutputExactGP, MultiOutputSparseGP, MultiOutputStochasticVariationalGP
from dotenv import load_dotenv
import os, time
load_dotenv()
home = os.getenv("MY_WS_HOME")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_data = np.zeros((10000, 2))
x_data[:, 0] = np.random.uniform(0, 1, 10000)
y_data = np.zeros((10000, 1))
# Define a noise standard deviation that varies smoothly with x
# noise_std = 0.2 + 1.0 * np.exp(-((x_data[:, 0] - 0.2) ** 2) / 0.01) + 0.8 * np.exp(-((x_data[:, 0] - 0.35) ** 2) / 0.005)
# noise_std = 0.2 + 0.8 * x_data[:, 0]  # Noise increases with x
noise_std = np.zeros_like(x_data[:, 0])
noise_std[x_data[:, 0] < 0.33] = 0.1
noise_std[(x_data[:, 0] > 0.33) & (x_data[:, 0] < 0.67)] = 0.4
noise_std[x_data[:, 0] > 0.67] = 0.7
y_data[:, 0] = np.sin(2 * np.pi * x_data[:, 0]) + np.random.normal(0, noise_std)

x_test = np.zeros((400, 2))
x_test[:, 0] = np.linspace(0, 1, 400)

X_train = torch.tensor(x_data, dtype=torch.float32).to(device)
Y_train = torch.tensor(y_data, dtype=torch.float32).to(device)

X_test = torch.tensor(x_test, dtype=torch.float32).to(device)

# Standard Multi-Output GP
models = {
    'multioutput': MultiOutputExactGP(
            X_train=X_train,
            Y_train=Y_train,
            device=device
    ),
    'sparse': MultiOutputSparseGP(
            input_dim=X_train.shape[1],
            output_dim=Y_train.shape[1],
            num_latents=Y_train.shape[1],
            independent=True,
            num_inducing_points=128,
            device=device
        ),
    'stochastic_variational': MultiOutputStochasticVariationalGP(
            input_dim=X_train.shape[1],
            output_dim=Y_train.shape[1],
            num_latents=Y_train.shape[1],
            independent=True,
            num_inducing_points=128,
            device=device
        )
    # 'sparse_heteroskedastic': MultiOutputSparseHeteroskedasticGP(
    #         input_dim=X_train.shape[1],
    #         output_dim=Y_train.shape[1],
    #         num_latents=Y_train.shape[1],
    #         num_inducing_points=128,
    #         device=device
    #     )
}

for name, model in models.items():
    print(f"Training {name} model...")
    if name == 'multioutput':
        start_time = time.time()
        model.train(X_train, Y_train, training_iter=100, lr=0.1)
        end_time = time.time()
    else:
        model.train(X_train, Y_train, num_epochs=100, lr=0.1)
        
results = {}
for name, model in models.items():
    Y_pred, Y_std, Y_lower, Y_upper = model.predict(X_test)
    results[name] = (
        Y_pred.cpu().numpy(),
        Y_std.cpu().numpy(),
        Y_lower.cpu().numpy(),
        Y_upper.cpu().numpy()
    )
    # print(f"{name} - Predicted mean: {Y_pred.mean().item():.2f}, Std: {Y_std.mean().item():.2f}")
    
# Plot results
plt.figure(figsize=(18, 5))
for i, (name, (Y_pred, Y_std, Y_lower, Y_upper)) in enumerate(results.items()):
    plt.subplot(1, 3, i + 1)
    plt.title(f"{name} GP")
    plt.fill_between(x_test[:, 0], Y_lower[:, 0], Y_upper[:, 0], alpha=0.3, label='Confidence Interval')
    plt.plot(x_test[:, 0], Y_pred[:, 0], 'r-', label='Prediction')
    plt.scatter(x_data[:, 0], y_data[:, 0], s=10, label='Training Data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.ylim(-3, 3)
    plt.legend()
    # print(f"{name} - Mean: {Y_pred.mean():.2f}, Std: {Y_std.mean():.2f}, Lower: {Y_lower.mean():.2f}, Upper: {Y_upper.mean():.2f}")
plt.tight_layout()
plt.savefig(f"{home}src/gp-ws/evaluate_out/simple_gp_validate_compare.png")
print("Done")
# plt.show()