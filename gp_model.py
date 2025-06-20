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
import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.kernels import RBFKernel, ScaleKernel, LinearKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from tqdm import tqdm
# import logging 
# from utils.logger import setup_logger, loss

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
        for i, (model, likelihood) in enumerate(zip(self.models, self.likelihoods)):
            model.eval()
            likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                pred = likelihood(model(X_test))
                preds.append(pred.mean.cpu().unsqueeze(-1))
                stds.append(pred.stddev.cpu().unsqueeze(-1))
        return torch.cat(preds, dim=-1), torch.cat(stds, dim=-1)


class SparseGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = (
            ScaleKernel(RBFKernel()) +
            ScaleKernel(gpytorch.kernels.LinearKernel())
        )
        # self.covar_module = ScaleKernel(RBFKernel()) 

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class MultiOutputSparseGP:
    def __init__(self, X_train, Y_train, num_inducing=128, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = []
        self.likelihoods = []
        self.X_train = X_train.to(self.device)
        self.Y_train = Y_train.to(self.device)

        for i in tqdm(range(Y_train.shape[-1]), desc="Initializing Sparse GPs"):
            inducing_points = self.X_train[:num_inducing].clone()
            model = SparseGPModel(inducing_points.to(self.device)).to(self.device)
            likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
            self.models.append(model)
            self.likelihoods.append(likelihood)
            
        print(f"Initialized {len(self.models)} Sparse GPs on device {self.device}")

    def train(self, num_epochs=None, batch_size=512, lr=0.01, logger=None):
        self.models = [m.train() for m in self.models]
        self.likelihoods = [l.train() for l in self.likelihoods]
        
        if num_epochs is None:
            num_epochs = [500, 500, 500, 100]

        for i, (model, likelihood) in enumerate(zip(self.models, self.likelihoods)):
            if logger:
                logger.info(f"Training Sparse GP {i+1}/{len(self.models)}")
            else:
                print(f"Training Sparse GP {i+1}/{len(self.models)}")
            optimizer = torch.optim.Adam([
                {'params': model.parameters()},
                {'params': likelihood.parameters()},
            ], lr=lr)

            mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=self.X_train.size(0))

            train_dataset = torch.utils.data.TensorDataset(self.X_train, self.Y_train[:, i])
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            for epoch in tqdm(range(num_epochs[i]), desc=f"Training Sparse GP {i+1}"):
                for x_batch, y_batch in train_loader:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    optimizer.zero_grad()
                    output = model(x_batch)
                    loss = -mll(output, y_batch)
                    if logger is not None:
                        logger.debug(f"Epoch {epoch+1} - Loss: {loss:.4f}")
                    loss.backward()
                    optimizer.step()

    def predict(self, X_test):
        X_test = X_test.to(self.device)
        preds = []
        stds = []
        for i, (model, likelihood) in enumerate(zip(self.models, self.likelihoods)):
            model.eval()
            likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                pred = likelihood(model(X_test))
                preds.append(pred.mean.cpu().unsqueeze(-1))
                stds.append(pred.stddev.cpu().unsqueeze(-1))
        return torch.cat(preds, dim=-1), torch.cat(stds, dim=-1)
    
class StochasticVariationalGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = (
            ScaleKernel(RBFKernel()) +
            ScaleKernel(LinearKernel())
        )
        # self.covar_module = ScaleKernel(RBFKernel()) 

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class MultiOutputStochasticVariationalGP:
    def __init__(self, X_train, Y_train, num_inducing=128, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = []
        self.likelihoods = []
        self.X_train = X_train.to(self.device)
        self.Y_train = Y_train.to(self.device)

        for i in tqdm(range(Y_train.shape[-1]), desc="Initializing Stochastic Variational GPs"):
            inducing_points = self.X_train[:num_inducing].clone()
            model = StochasticVariationalGP(inducing_points.to(self.device)).to(self.device)
            likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
            self.models.append(model)
            self.likelihoods.append(likelihood)
            
        print(f"Initialized {len(self.models)} Stochastic Variational GPs on device {self.device}")

    def train(self, num_epochs=None, batch_size=512, lr=0.01, logger=None):
        self.models = [m.train() for m in self.models]
        self.likelihoods = [l.train() for l in self.likelihoods]
        
        if num_epochs is None:
            num_epochs = [500, 500, 500, 100]

        for i, (model, likelihood) in enumerate(zip(self.models, self.likelihoods)):
            if logger:
                logger.info(f"Training Stochastic Variational GP {i+1}/{len(self.models)}")
            else:
                print(f"Training Stochastic Variational GP {i+1}/{len(self.models)}")
            optimizer = torch.optim.Adam([
                {'params': model.parameters()},
                {'params': likelihood.parameters()},
            ], lr=lr)

            mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=self.X_train.size(0))

            train_dataset = torch.utils.data.TensorDataset(self.X_train, self.Y_train[:, i])
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            for _ in tqdm(range(num_epochs[i]), desc=f"Training Stochastic Variational GP {i+1}"):
                for x_batch, y_batch in train_loader:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    optimizer.zero_grad()
                    output = model(x_batch)
                    loss = -mll(output, y_batch)
                    if logger is not None:
                        logger.debug(f"Epoch {_+1} - Loss: {loss:.4f}")
                    loss.backward()
                    optimizer.step()

    def predict(self, X_test):
        X_test = X_test.to(self.device)
        preds = []
        stds = []
        for i, (model, likelihood) in enumerate(zip(self.models, self.likelihoods
        )):
            model.eval()
            likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                pred = likelihood(model(X_test))
                preds.append(pred.mean.cpu().unsqueeze(-1))
                stds.append(pred.stddev.cpu().unsqueeze(-1))
        return torch.cat(preds, dim=-1), torch.cat(stds, dim=-1)