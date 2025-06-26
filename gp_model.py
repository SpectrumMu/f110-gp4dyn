"""
This module provides classes for multi-output Gaussian Process (GP) regression using GPyTorch, supporting both exact and sparse (variational) inference.
Classes:
--------
SingleOutputGPModel(ExactGP):
    A single-output exact GP regression model with zero mean and RBF kernel.
MultiOutputGP:
    Handles multi-output regression by training one SingleOutputGPModel per output dimension.
    Args:
        X_train (torch.Tensor): Training inputs of shape (N, D).
        Y_train (torch.Tensor): Training targets of shape (N, M), where M is the number of outputs.
        device (str, optional): Device to use ('cuda' or 'cpu'). Defaults to CUDA if available.
    Methods:
        train(X_train, Y_train, training_iter=100):
            Trains each GP model for the specified number of iterations.
        predict(X_test):
            Returns mean predictions for each output on the test set.
SparseGPModel(gpytorch.models.ApproximateGP):
    A single-output sparse (variational) GP regression model with zero mean and RBF kernel.
    Args:
        inducing_points (torch.Tensor): Initial inducing points for variational inference.
MultiOutputSparseGP:
    Handles multi-output regression using sparse GP models (one per output dimension).
    Args:
        X_train (torch.Tensor): Training inputs of shape (N, D).
        Y_train (torch.Tensor): Training targets of shape (N, M).
        num_inducing (int): Number of inducing points per output GP.
        device (str, optional): Device to use ('cuda' or 'cpu'). Defaults to CUDA if available.
    Methods:
        train(num_epochs=100, batch_size=512):
            Trains each sparse GP model using mini-batch variational inference.
        predict(X_test):
            Returns mean predictions for each output on the test set.
Notes:
------
- All models use a zero mean function and an RBF kernel with automatic relevance determination (ARD).
- Multi-output is handled by independent GPs for each output dimension.
- Training and prediction are performed on the specified device (CPU or GPU).
"""
from pyexpat import model
import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.kernels import RBFKernel, ScaleKernel, LinearKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from tqdm import tqdm
# import logging 
# from utils.logger import setup_logger, loss

gpytorch.settings.cholesky_jitter.default_value = 1e-3

class SingleOutputGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class MultiOutputGP:
    def __init__(self, X_train, Y_train, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = []
        self.likelihoods = []
        self.mlls = []

        X_train = X_train.to(self.device)
        Y_train = Y_train.to(self.device)
        
        for i in tqdm(range(Y_train.shape[-1]), desc="Initializing GPs"):
            y = Y_train[:, i]
            likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
            model = SingleOutputGPModel(X_train, y, likelihood).to(self.device)
            self.models.append(model)
            self.likelihoods.append(likelihood)
            self.mlls.append(gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model))

        print(f"Initialized {len(self.models)} GPs on device {self.device}")

    def train(self, X_train, Y_train, training_iter=100, lr=0.01, logger=None):
        X_train = X_train.to(self.device)
        Y_train = Y_train.to(self.device)

        for i, model in enumerate(self.models):
            if logger:
                logger.info(f"Training GP {i+1}/{len(self.models)}")
            else:
                print(f"Training GP {i+1}/{len(self.models)}")
            model.train()
            self.likelihoods[i].train()

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            mll = self.mlls[i]

            for _ in tqdm(range(training_iter), desc=f"Training GP {i+1}"):
                optimizer.zero_grad()
                output = model(X_train)
                loss = -mll(output, Y_train[:, i])
                if logger is not None:
                    logger.debug(f"GP {i+1} Loss: {loss.item()}")
                loss.backward()
                optimizer.step()

    def predict(self, X_test):
        X_test = X_test.to(self.device)
        preds = []
        stds = []
        lower, upper = [], []
        for i, (model, likelihood) in enumerate(zip(self.models, self.likelihoods)):
            model.eval()
            likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                pred = likelihood(model(X_test))
                preds.append(pred.mean.cpu().unsqueeze(-1))
                stds.append(pred.stddev.cpu().unsqueeze(-1))
                lower.append(pred.confidence_region()[0].cpu().unsqueeze(-1))
                upper.append(pred.confidence_region()[1].cpu().unsqueeze(-1))
        return torch.cat(preds, dim=-1), torch.cat(stds, dim=-1), torch.cat(lower, dim=-1), torch.cat(upper, dim=-1)


class SparseGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, input_dim, output_dim, num_latents, independent=False, num_inducing_points=256):
        # Initialize inducing points for each latent function
        inducing_points = torch.rand(num_latents, num_inducing_points, input_dim)
        
        # Variational distribution with batch shape for each latent
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_latents])
        )
        
        # Choose variational strategy
        if independent:
            variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
                gpytorch.variational.VariationalStrategy(
                    self, inducing_points, variational_distribution, learn_inducing_locations=True
                ),
                num_tasks=output_dim
                # latent_dim=-1,
            )
        else:
            variational_strategy = gpytorch.variational.LMCVariationalStrategy(
                gpytorch.variational.VariationalStrategy(
                    self, inducing_points, variational_distribution, learn_inducing_locations=True
                ),
                num_tasks=output_dim,
                num_latents=num_latents,
                latent_dim=-1
            )
        
        super().__init__(variational_strategy)
        
        # Mean and covariance modules with batch shape for each latent
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                # nu=2.5,  # Matern kernel with nu=2.5
                ard_num_dims=inducing_points.size(2),
                batch_shape=torch.Size([num_latents])
            )
            + gpytorch.kernels.RBFKernel(
                ard_num_dims=inducing_points.size(2),
                batch_shape=torch.Size([num_latents])
            )   
        )
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class MultiOutputSparseGP:
    def __init__(self, input_dim, output_dim, num_latents, independent=False, num_inducing_points=256, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SparseGPModel(input_dim, output_dim, num_latents, independent, num_inducing_points).to(self.device)
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=output_dim).to(self.device)
        print(f"Initialized MultiOutputSparseGP on device {self.device}")

    def train(self, X_train, Y_train, num_epochs=100, batch_size=512, lr=0.01, logger=None):
        self.model.train()
        self.likelihood.train()

        X_train = X_train.to(self.device)
        Y_train = Y_train.to(self.device)

        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.likelihood.parameters()},
        ], lr=lr)

        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=X_train.size(0))

        train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        for epoch in tqdm(range(num_epochs), desc="Training MultiOutputSparseGP"):
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                output = self.model(x_batch)
                loss = -mll(output, y_batch)
                if logger:
                    logger.debug(f"Epoch {epoch+1} - Loss: {loss.item():.4f}")
                loss.backward()
                optimizer.step()

    def predict(self, X_test):
        self.model.eval()
        self.likelihood.eval()

        X_test = X_test.to(self.device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.model(X_test))
            preds = pred.mean.cpu()
            stds = pred.stddev.cpu()
            lower, upper = pred.confidence_region()
            lower = lower.cpu()
            upper = upper.cpu()

        return preds, stds, lower, upper

class StochasticVariationalGP(gpytorch.models.ApproximateGP):
    def __init__(self, input_dim, output_dim, num_latents, independent=False, num_inducing_points=256):
        # Different inducing points for each latent function
        inducing_points = torch.rand(num_latents, num_inducing_points, input_dim)
        
        # Mark the variational distribution as batch to learn one per latent
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_latents])
        )
        
        variational_strategy = None
        if independent:
            variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
                gpytorch.variational.VariationalStrategy(
                    self, inducing_points, variational_distribution, learn_inducing_locations=True
                ),
                num_tasks=output_dim,
                # latent_dim=-1,
            )
        else:
            # Wrap the VariationalStrategy in an LMCVariationalStrategy
            variational_strategy = gpytorch.variational.LMCVariationalStrategy(
                gpytorch.variational.VariationalStrategy(
                    self, inducing_points, variational_distribution, learn_inducing_locations=True
                ),
                num_tasks=output_dim,
                num_latents=num_latents,
                latent_dim=-1
            )
        
        super().__init__(variational_strategy)
        
        # Mean and covariance functions are set up with batch shape for each latent
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latents])),
            batch_shape=torch.Size([num_latents])
        )
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class MultiOutputStochasticVariationalGP:
    def __init__(self, input_dim, output_dim, num_latents, independent=False, num_inducing_points=256, device=None):
        if device is not None:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = StochasticVariationalGP(input_dim, output_dim, num_latents, independent, num_inducing_points)
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=output_dim)

        print(f"Initialized Stochastic Variational GP on device {self.device}")

    def train(self, num_epochs=None, batch_size=512, lr=0.01, logger=None):
        self.model.train()
        self.likelihood.train()

        if logger:
            logger.info(f"Training Stochastic Variational GP {i+1}/{len(self.models)}")
        else:
            print(f"Training Stochastic Variational GP {i+1}/{len(self.models)}")
        optimizer = torch.optim.Adam([
                {'params': self.model.parameters()},
                {'params': self.likelihood.parameters()},
        ], lr=lr)

        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=self.X_train.size(0))

        train_dataset = torch.utils.data.TensorDataset(self.X_train, self.Y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        for _ in tqdm(range(num_epochs), desc="Training Stochastic Variational GP"):
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                output = self.model(x_batch)
                loss = -mll(output, y_batch)
                if logger is not None:
                    logger.debug(f"Epoch {_+1} - Loss: {loss:.4f}")
                    loss.backward()
                    optimizer.step()

    def predict(self, X_test):
        X_test = X_test.to(self.device)
        preds = None
        stds = None
        lower, upper = None
        
        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.model(X_test))
            preds = pred.mean.cpu().unsqueeze(-1)
            stds = pred.stddev.cpu().unsqueeze(-1)
            lower = pred.confidence_region()[0].cpu().unsqueeze(-1)
            upper = pred.confidence_region()[1].cpu().unsqueeze(-1)
        
        return preds, stds, lower, upper