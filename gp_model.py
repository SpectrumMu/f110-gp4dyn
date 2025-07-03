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

# class SingleOutputGPModel(ExactGP):
#     def __init__(self, train_x, train_y, likelihood):
#         super().__init__(train_x, train_y, likelihood)
#         self.mean_module = gpytorch.means.ZeroMean()
#         self.covar_module = ScaleKernel(RBFKernel())

#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# class MultiOutputGP:
#     def __init__(self, X_train, Y_train, device=None):
#         self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
#         self.models = []
#         self.likelihoods = []
#         self.mlls = []

#         X_train = X_train.to(self.device)
#         Y_train = Y_train.to(self.device)
        
#         for i in tqdm(range(Y_train.shape[-1]), desc="Initializing GPs"):
#             y = Y_train[:, i]
#             likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
#             model = SingleOutputGPModel(X_train, y, likelihood).to(self.device)
#             self.models.append(model)
#             self.likelihoods.append(likelihood)
#             self.mlls.append(gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model))

#         print(f"Initialized {len(self.models)} GPs on device {self.device}")

#     def train(self, X_train, Y_train, training_iter=100, lr=0.01, logger=None):
#         X_train = X_train.to(self.device)
#         Y_train = Y_train.to(self.device)

#         for i, model in enumerate(self.models):
#             if logger:
#                 logger.info(f"Training GP {i+1}/{len(self.models)}")
#             else:
#                 print(f"Training GP {i+1}/{len(self.models)}")
#             model.train()
#             self.likelihoods[i].train()

#             optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#             mll = self.mlls[i]

#             for _ in tqdm(range(training_iter), desc=f"Training GP {i+1}"):
#                 optimizer.zero_grad()
#                 output = model(X_train)
#                 loss = -mll(output, Y_train[:, i])
#                 if logger is not None:
#                     logger.debug(f"GP {i+1} Loss: {loss.item()}")
#                 loss.backward()
#                 optimizer.step()

#     def predict(self, X_test):
#         X_test = X_test.to(self.device)
#         preds = []
#         stds = []
#         lower, upper = [], []
#         for i, (model, likelihood) in enumerate(zip(self.models, self.likelihoods)):
#             model.eval()
#             likelihood.eval()
#             with torch.no_grad(), gpytorch.settings.fast_pred_var():
#                 pred = likelihood(model(X_test))
#                 preds.append(pred.mean.cpu().unsqueeze(-1))
#                 stds.append(pred.stddev.cpu().unsqueeze(-1))
#                 lower.append(pred.confidence_region()[0].cpu().unsqueeze(-1))
#                 upper.append(pred.confidence_region()[1].cpu().unsqueeze(-1))
#         return torch.cat(preds, dim=-1), torch.cat(stds, dim=-1), torch.cat(lower, dim=-1), torch.cat(upper, dim=-1)

class ExactMIMOGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, output_dim):
        # Transpose Y_train to match batch shape expectations
        if train_y.dim() == 2 and train_y.shape[1] == output_dim:
            train_y = train_y.t()  # Shape: [output_dim, num_data]
        
        super().__init__(train_x, train_y, likelihood)
        self.output_dim = output_dim
        
        # Use batch shape for multi-output
        self.mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([output_dim]))
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=torch.Size([output_dim])),
            batch_shape=torch.Size([output_dim])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class MultiOutputExactGP:
    def __init__(self, X_train, Y_train, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        X_train = X_train.to(self.device)
        Y_train = Y_train.to(self.device)
        
        self.output_dim = Y_train.shape[-1]
        
        # Use regular GaussianLikelihood instead of MultitaskGaussianLikelihood
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            batch_shape=torch.Size([self.output_dim])
        ).to(self.device)
        
        self.model = ExactMIMOGPModel(
            X_train, Y_train, self.likelihood, self.output_dim
        ).to(self.device)
        
        print(f"Initialized ExactMIMOGP with {self.output_dim} outputs on device {self.device}")

    def train(self, X_train, Y_train, training_iter=100, lr=0.01, logger=None):
        self.model.train()
        self.likelihood.train()
        
        X_train = X_train.to(self.device)
        Y_train = Y_train.to(self.device)
        
        # Transpose Y_train if needed for batch processing
        if Y_train.dim() == 2 and Y_train.shape[1] == self.output_dim:
            Y_train = Y_train.t()  # Shape: [output_dim, num_data]

        all_params = set(self.model.parameters()) | set(self.likelihood.parameters())
        optimizer = torch.optim.Adam(list(all_params), lr=lr)
        
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        best_loss = float('inf')
        patience_counter = 0
        patience_limit = 50

        for epoch in tqdm(range(training_iter), desc="Training ExactMIMOGP"):
            optimizer.zero_grad()
            output = self.model(X_train)
            loss = -mll(output, Y_train)
            
            # Ensure loss is scalar by taking mean if it's a tensor
            if loss.numel() > 1:
                loss = loss.mean()
            
            if logger is not None:
                logger.debug(f"Epoch {epoch+1} Loss: {loss.item():.4f}")
            
            loss.backward()
            optimizer.step()
            
            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience_limit:
                if logger:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                break

    def predict(self, X_test):
        self.model.eval()
        self.likelihood.eval()
        
        X_test = X_test.to(self.device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.model(X_test))
            preds = pred.mean.cpu().t()  # Transpose back to [num_test, output_dim]
            stds = pred.stddev.cpu().t()
            lower, upper = pred.confidence_region()
            lower = lower.cpu().t()
            upper = upper.cpu().t()
            
        return preds, stds, lower, upper

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
            gpytorch.kernels.MaternKernel(
                nu=2.5,  # Change from 0.5 to 2.5 for smoother functions
                ard_num_dims=inducing_points.size(2),
                batch_shape=torch.Size([num_latents])
            )
            # + 
            # gpytorch.kernels.RBFKernel(
            #     ard_num_dims=inducing_points.size(2),
            #     batch_shape=torch.Size([num_latents])
            # )   
            # # Add linear kernel for dynamics modeling
            # +
            # gpytorch.kernels.LinearKernel(
            #     ard_num_dims=inducing_points.size(2),
            #     batch_shape=torch.Size([num_latents])
            # )
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
        
        # Add learning rate scheduler for adaptive learning rate
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=20
        )

        mll = gpytorch.mlls.PredictiveLogLikelihood(self.likelihood, self.model, num_data=X_train.size(0))

        train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        best_loss = float('inf')
        patience_counter = 0
        patience_limit = 50
        
        for epoch in tqdm(range(num_epochs), desc="Training MultiOutputSparseGP"):
            epoch_loss = 0.0
            batch_count = 0
            
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                output = self.model(x_batch)
                loss = -mll(output, y_batch)
                epoch_loss += loss.item()
                batch_count += 1
                
                if logger is not None:
                    logger.debug(f"Epoch {epoch+1} - Batch Loss: {loss.item():.4f}")

                loss.backward()
                optimizer.step()
            
            avg_epoch_loss = epoch_loss / batch_count
            scheduler.step(avg_epoch_loss)
            
            if logger:
                logger.debug(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_epoch_loss:.4f}")
            
            # Early stopping
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience_limit:
                if logger:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                break

    def predict(self, X_test):
        self.model.eval()
        self.likelihood.eval()

        X_test = X_test.to(self.device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.model(X_test))
            preds = pred.mean.cpu()
            stds = pred.stddev.exp().cpu()
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
            gpytorch.kernels.MaternKernel(
                nu=2.5,
                ard_num_dims=inducing_points.size(2),
                batch_shape=torch.Size([num_latents])
            )
        )
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class MultiOutputStochasticVariationalGP:
    def __init__(self, input_dim, output_dim, num_latents, independent=False, num_inducing_points=256, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = StochasticVariationalGP(input_dim, output_dim, num_latents, independent, num_inducing_points).to(self.device)
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=output_dim).to(self.device)

        print(f"Initialized Stochastic Variational GP on device {self.device}")

    def train(self, X_train, Y_train, num_epochs=None, batch_size=512, lr=0.01, logger=None):
        self.model.train()
        self.likelihood.train()

        X_train = X_train.to(self.device)
        Y_train = Y_train.to(self.device)

        optimizer = torch.optim.Adam([
                {'params': self.model.parameters()},
                {'params': self.likelihood.parameters()},
        ], lr=lr)

        mll = gpytorch.mlls.PredictiveLogLikelihood(self.likelihood, self.model, num_data=X_train.size(0))

        train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
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
        self.model.eval()
        self.likelihood.eval()

        X_test = X_test.to(self.device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.model(X_test))
            preds = pred.mean.cpu()
            stds = pred.stddev.exp().cpu()
            lower, upper = pred.confidence_region()
            lower = lower.cpu()
            upper = upper.cpu()
        
        return preds, stds, lower, upper

class SparseHeteroskedasticGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, input_dim, output_dim, num_latents, num_inducing_points=256):
        # Main GP inducing points
        inducing_points = torch.rand(num_latents, num_inducing_points, input_dim)
        
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_latents])
        )
        
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=output_dim,
            num_latents=num_latents,
            latent_dim=-1
        )
        
        super().__init__(variational_strategy)
        
        # Main GP components
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=2.5,
                ard_num_dims=input_dim,
                batch_shape=torch.Size([num_latents])
            ),
            batch_shape=torch.Size([num_latents])
        )
        
        # Separate GP for noise modeling
        noise_inducing_points = torch.rand(num_inducing_points // 4, input_dim)
        self.noise_gp = gpytorch.models.ApproximateGP(
            gpytorch.variational.VariationalStrategy(
                self,
                noise_inducing_points,
                gpytorch.variational.CholeskyVariationalDistribution(num_inducing_points // 4),
                learn_inducing_locations=True
            )
        )
        
        # Noise GP components (simpler, single output)
        self.noise_mean = gpytorch.means.ConstantMean()
        self.noise_covar = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=input_dim))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def noise_forward(self, x):
        # Simple noise model - predict log noise variance
        noise_mean = self.noise_mean(x)
        noise_covar = self.noise_covar(x)
        noise_dist = gpytorch.distributions.MultivariateNormal(noise_mean, noise_covar)
        # Return log noise to ensure positivity when exponentiated
        return noise_dist

class MultiOutputSparseHeteroskedasticGP:
    def __init__(self, input_dim, output_dim, num_latents, num_inducing_points=256, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dim = output_dim
        
        self.model = SparseHeteroskedasticGPModel(
            input_dim, output_dim, num_latents, num_inducing_points
        ).to(self.device)
        
        # Start with fixed noise likelihood
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=output_dim,
            noise_constraint=gpytorch.constraints.GreaterThan(1e-6)
        ).to(self.device)
        
        print(f"Initialized Sparse Heteroskedastic GP with {output_dim} outputs on device {self.device}")

    def train(self, X_train, Y_train, num_epochs=100, batch_size=512, lr=0.01, logger=None):
        self.model.train()
        self.likelihood.train()

        X_train = X_train.to(self.device)
        Y_train = Y_train.to(self.device)

        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.likelihood.parameters()},
        ], lr=lr)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=15
        )

        # Use custom loss that combines main GP and noise GP
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=X_train.size(0))

        train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        best_loss = float('inf')
        patience_counter = 0
        patience_limit = 40
        
        for epoch in tqdm(range(num_epochs), desc="Training Sparse Heteroskedastic GP"):
            epoch_loss = 0.0
            batch_count = 0
            
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                
                # Main GP prediction
                f_pred = self.model(x_batch)
                
                # Noise GP prediction (log noise variance)
                noise_pred = self.model.noise_forward(x_batch)
                log_noise_var = noise_pred.mean  # Use mean prediction
                noise_var = torch.exp(log_noise_var).clamp(min=1e-6)
                
                # Custom heteroskedastic loss
                # Negative log likelihood with learned noise
                residuals = (y_batch.squeeze() - f_pred.mean.t()).pow(2)
                het_loss = 0.5 * (residuals / noise_var + torch.log(noise_var)).mean()
                
                # Add GP prior terms
                main_loss = -mll(f_pred, y_batch)
                noise_prior_loss = -noise_pred.log_prob(torch.zeros_like(noise_pred.mean)).mean()
                
                # Combine losses
                total_loss = main_loss + het_loss + 0.01 * noise_prior_loss

                epoch_loss += total_loss.item()
                batch_count += 1
                
                if logger is not None:
                    logger.debug(f"Epoch {epoch+1} - Batch Loss: {total_loss.item():.4f}")

                total_loss.backward()
                optimizer.step()
            
            avg_epoch_loss = epoch_loss / batch_count
            scheduler.step(avg_epoch_loss)
            
            if logger:
                logger.debug(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_epoch_loss:.4f}")
            
            # Early stopping
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience_limit:
                if logger:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                break

    def predict(self, X_test):
        self.model.eval()
        self.likelihood.eval()

        X_test = X_test.to(self.device)
        
        with torch.no_grad():
            # Main GP predictions
            f_pred = self.model(X_test)
            f_mean = f_pred.mean.t()  # Shape: [n_test, n_outputs]
            f_var = f_pred.variance.t()
            
            # Noise predictions
            noise_pred = self.model.noise_forward(X_test)
            log_noise_var = noise_pred.mean
            noise_var = torch.exp(log_noise_var).clamp(min=1e-6)
            
            # Fix shape mismatch - ensure noise_var matches f_var dimensions
            if noise_var.dim() == 1:
                noise_var = noise_var.unsqueeze(-1)  # Shape: [n_test, 1]
            
            if noise_var.shape != f_var.shape:
                noise_var = noise_var.t()  # Ensure shape matches f_var

            noise_var = noise_var.expand_as(f_var)
            
            # Total predictive variance = GP variance + learned noise
            total_var = f_var + noise_var
            total_std = total_var.sqrt()
            
            # Confidence intervals using total uncertainty
            lower = f_mean - 1.96 * total_std
            upper = f_mean + 1.96 * total_std

        # Ensure the output shape matches [n_test, n_outputs]
        return f_mean.t().cpu(), total_std.t().cpu(), lower.t().cpu(), upper.t().cpu()

    def predict_with_noise(self, X_test):
        """Predict both mean function and noise function separately"""
        self.model.eval()
        X_test = X_test.to(self.device)
        
        with torch.no_grad():
            # Main GP predictions
            f_pred = self.model(X_test)
            main_mean = f_pred.mean.cpu()
            main_var = f_pred.variance.cpu()
            
            # Noise GP predictions
            noise_pred = self.model.noise_forward(X_test)
            log_noise_var = noise_pred.mean.cpu()
            noise_var = torch.exp(log_noise_var)
            
        return main_mean, main_var, noise_var

