import torch
import gpytorch

class MIMO_SVGPModel(gpytorch.models.ApproximateGP):
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
                latent_dim=-1,
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
    
class MIMO_SVGP():
    def __init__(self, input_dim, output_dim, num_latents, independent=False, num_inducing_points=256):
        self.model = MIMO_SVGPModel(input_dim, output_dim, num_latents, independent, num_inducing_points)
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=output_dim)
    
    def train(self):
        self.model.train()
        self.likelihood.train()
    
    def eval(self):
        self.model.eval()
        self.likelihood.eval()

    def predict(self, x):
        """
        Predict the output for the given input x.
        Args:
            x (torch.Tensor): Input tensor of shape (N, input_dim) where N is the number of samples.
        Returns:
            torch.Tensor: Predicted output tensor of shape (N, output_dim).
        """
        preds = self.model(x)
        return self.likelihood(preds).rsample()