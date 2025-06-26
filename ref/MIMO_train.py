import math
import torch
import gpytorch
import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from MIMO_SVGP import MIMO_SVGP

# -------------------------------------------------------------------
# Generate training data on a grid
xv, yv = torch.meshgrid(
    torch.linspace(0, 1, 100),
    torch.linspace(0, 1, 100)
)
train_x = torch.stack([xv.flatten(), yv.flatten()], -1)

train_y = torch.stack([
    torch.sin(train_x[:, 0] * (2 * math.pi)) + 2 * torch.cos(train_x[:, 1] * (1 * math.pi))  + torch.randn(train_x[:, 0].size()) * 0.2,
    torch.cos(train_x[:, 0] * (2 * math.pi)) + 2 * torch.cos(train_x[:, 1] * (3/2 * math.pi))  + torch.randn(train_x[:, 0].size()) * 0.2,
    torch.sin(train_x[:, 0] * (2 * math.pi)) + 2 * torch.cos(train_x[:, 1] * (2 * math.pi))      + torch.randn(train_x[:, 0].size()) * 0.2,
    -torch.cos(train_x[:, 0] * (2 * math.pi)) + 2 * torch.cos(train_x[:, 1] * (1/2 * math.pi))   + torch.randn(train_x[:, 0].size()) * 0.2,
], -1)

print(train_x.shape, train_y.shape)

num_latents = 3
num_tasks = train_y.shape[1]

# Create model and likelihood
svgp = MIMO_SVGP(
    input_dim=train_x.shape[1],
    output_dim=train_y.shape[1],
    num_latents=num_latents,
    independent=False,
    num_inducing_points=16
)
print(svgp.predict(train_x).rsample().shape)

# -------------------------------------------------------------------
# (Optional) An alternative independent multitask GP model is defined below.
# model = MIMO_SVGPModel(
#     input_dim=train_x.shape[1],
#     output_dim=train_y.shape[1],
#     num_latents=num_latents,
#     independent=True,
#     num_inducing_points=16
# )

# -------------------------------------------------------------------
# Training settings
import os
import time
from mpl_toolkits.mplot3d import Axes3D  # (for compatibility)
smoke_test = ('CI' in os.environ)
# smoke_test = True
num_epochs = 1 if smoke_test else 500
svgp.train()

optimizer = torch.optim.Adam([
    {'params': svgp.model.parameters()},
    {'params': svgp.likelihood.parameters()},
], lr=0.1)

mll = gpytorch.mlls.VariationalELBO(svgp.likelihood, svgp.model, num_data=train_y.size(0))

epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
for i in epochs_iter:
    optimizer.zero_grad()
    output = svgp.model(train_x)
    loss = -mll(output, train_y)
    epochs_iter.set_postfix(loss=loss.item())
    loss.backward()
    optimizer.step()

# Set into evaluation mode
svgp.eval()

# -------------------------------------------------------------------
# Create a Plotly figure with 1 row and 4 columns (one per task)
fig = make_subplots(
    rows=1, cols=4,
    specs=[[{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]],
    subplot_titles=[f"Task {i+1}" for i in range(num_tasks)],
    horizontal_spacing=0.01,  # Minimal horizontal spacing between subplots
    vertical_spacing=0.01,    # Minimal vertical spacing between subplots
    
)

# -------------------------------------------------------------------
# Make predictions on a test grid
start_time = time.time()
num_test_points = 100
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_xv, test_yv = torch.meshgrid(
        torch.linspace(0, 1, num_test_points),
        torch.linspace(0, 1, num_test_points)
    )
    test_x = torch.stack([test_xv.flatten(), test_yv.flatten()], -1)
    print(test_x.shape)
    predictions = svgp.predict(test_x)
    mean = predictions.mean       # shape: (num_test_points*num_test_points, num_tasks)
    lower, upper = predictions.confidence_region()

end_time = time.time()
print(f"Prediction block execution time: {end_time - start_time:.4f} seconds")

# Convert the grid to NumPy arrays
X_np = test_xv.numpy()
Y_np = test_yv.numpy()

# Convert training data to NumPy arrays
train_x_np = train_x.detach().cpu().numpy()
train_y_np = train_y.detach().cpu().numpy()

# -------------------------------------------------------------------
# Extract and predict over inducing points for visualization.
# Here we extract inducing points from the underlying variational strategy.
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    # For LMCVariationalStrategy, the inducing points are stored in:
    # model.variational_strategy.variational_strategy.inducing_points
    inducing_points = svgp.model.variational_strategy.base_variational_strategy.inducing_points.detach()
    # Reshape from [num_latents, num_inducing, 2] to [num_latents * num_inducing, 2]
    inducing_points_all = inducing_points.reshape(-1, 2)
    inducing_predictions = svgp.predict((inducing_points_all))
    inducing_mean = inducing_predictions.mean  # shape: (num_latents*num_inducing, num_tasks)
    
# Convert inducing points and predictions to NumPy arrays
inducing_points_np = inducing_points_all.cpu().numpy()
inducing_mean_np = inducing_mean.cpu().numpy()

# -------------------------------------------------------------------
# Loop over each task and add surfaces and scatter traces
for task in range(num_tasks):
    # Reshape predictions for plotting
    mean_reshaped = mean[:, task].view(num_test_points, num_test_points).detach().cpu().numpy()
    lower_reshaped = lower[:, task].view(num_test_points, num_test_points).detach().cpu().numpy()
    upper_reshaped = upper[:, task].view(num_test_points, num_test_points).detach().cpu().numpy()

    # Add the mean predictive surface
    fig.add_trace(
        go.Surface(
            x=X_np, y=Y_np, z=mean_reshaped,
            colorscale='Viridis',
            opacity=0.8,
            name='Mean',
            showscale=False
        ),
        row=1, col=task+1
    )
    # Add the lower confidence bound surface
    fig.add_trace(
        go.Surface(
            x=X_np, y=Y_np, z=lower_reshaped,
            colorscale='Greys',
            opacity=0.3,
            name='Lower Bound',
            showscale=False
        ),
        row=1, col=task+1
    )
    # Add the upper confidence bound surface
    fig.add_trace(
        go.Surface(
            x=X_np, y=Y_np, z=upper_reshaped,
            colorscale='Greys',
            opacity=0.3,
            name='Upper Bound',
            showscale=False
        ),
        row=1, col=task+1
    )
    # Overlay the training data as scattered points
    fig.add_trace(
        go.Scatter3d(
            x=train_x_np[:, 0],
            y=train_x_np[:, 1],
            z=train_y_np[:, task],
            mode='markers',
            marker=dict(color='black', size=0.5),
            name='Training Data'
        ),
        row=1, col=task+1
    )
    # Overlay the inducing points as red markers.
    # Their predicted mean for the current task is used for the z-axis.
    fig.add_trace(
        go.Scatter3d(
            x=inducing_points_np[:, 0],
            y=inducing_points_np[:, 1],
            z=inducing_mean_np[:, task],
            mode='markers',
            marker=dict(color='red', size=3, symbol='diamond'),
            name='Inducing Points'
        ),
        row=1, col=task+1
    )

# -------------------------------------------------------------------
# Update layout settings to maximize available space
fig.update_layout(
    title_text="Predictive Surfaces & Training Data per Task",
    autosize=True,
    margin=dict(l=0, r=0, t=30, b=0)
)

# Set axis titles and aspect ratio for each subplot scene
for i in range(1, num_tasks+1):
    scene_id = f'scene{i}'
    fig['layout'][scene_id].update(
        xaxis_title='X1',
        yaxis_title='X2',
        zaxis_title='Output',
        aspectratio=dict(x=1, y=1, z=1),
        camera_eye=dict(x=2.0, y=2.0, z=2.0)
    )

fig.show()
