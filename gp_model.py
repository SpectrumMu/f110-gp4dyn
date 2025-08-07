import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.kernels import RBFKernel, ScaleKernel, LinearKernel
from tqdm import tqdm
# import logging 
# from utils.logger import setup_logger, loss

gpytorch.settings.cholesky_jitter.default_value = 1e-3

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
        losses = []

        for epoch in tqdm(range(training_iter), desc="Training ExactMIMOGP"):
            optimizer.zero_grad()
            output = self.model(X_train)
            loss = -mll(output, Y_train)

            # Ensure loss is scalar by taking mean if it's a tensor
            if loss.numel() > 1:
                loss = loss.mean()
            
            if logger is not None:
                logger.debug(f"Epoch {epoch+1} Loss: {loss.item():.4f}")
            
            losses.append(loss.item())
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

        return losses

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
        epoch_losses = []
        
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
            epoch_losses.append(avg_epoch_loss)
            
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

        return epoch_losses

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

    def save(self, filepath):
        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad():
            torch.save(self.model.state_dict(), filepath + "model.pth")
            torch.save(self.likelihood.state_dict(), filepath + "likelihood.pth")

    def load(self, filepath):
        self.model.eval()
        self.likelihood.eval()

        self.model.load_state_dict(torch.load(filepath + "model.pth"))
        self.likelihood.load_state_dict(torch.load(filepath + "likelihood.pth"))

        self.model.to(self.device)
        self.likelihood.to(self.device)

        # print(f"Loaded MultiOutputSparseGP from {filepath}")

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

        losses = []

        for _ in tqdm(range(num_epochs), desc="Training Stochastic Variational GP"):
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

                # if logger is not None:
                #     logger.debug(f"Epoch {_+1} - Loss: {loss:.4f}")

                loss.backward()
                optimizer.step()

            epoch_loss /= batch_count
            losses.append(epoch_loss)

            if logger is not None:
                logger.debug(f"Epoch {_+1}/{num_epochs} - Avg Loss: {epoch_loss:.4f}")

        return losses

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

