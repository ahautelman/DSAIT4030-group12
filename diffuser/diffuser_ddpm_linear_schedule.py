import torch

class Diffuser_DDPM_linear_schedule:

    """
    Implementation of a Denoising Diffusion Probabilistic Model (DDPM) with a linear noise schedule.
    The structure of the module is inspired by the "Spectrum Matching: a Unified Perspective for Superior Diffusability in Latent Diffusion" implementation (/modules/scheduler.py on https://github.com/forever208/SpectrumMatching).
    DDPM formulations of the forward and reverse diffusion processes are based on the original DDPM paper (https://arxiv.org/abs/2006.11239).
    """

    def __init__(self, total_timesteps=1000, beta_start=0.0001, beta_end=0.02):

        # Number of diffusion steps T
        self.total_timesteps = total_timesteps
        # Start value of the noise schedule
        self.beta_start = beta_start
        # End value of the noise schedule
        self.beta_end = beta_end
        
        # The beta values define how much noise is added at each diffusion step, here represented as a linear interpolation between beta_start and beta_end.
        self.betas = self.determine_linear_beta_set()
        # The alpha values represent how much of the original signal is preserved at each diffusion step, hence calucalted as the differece between 1 and beta.
        self.alphas = torch.sub(1.0, self.betas)
        # The alpha bar values represent the cummalitive product of alpha values as alpha_0 * ... * alpha_t for each t in [0, T-1].
        self.alpha_bars = torch.cumprod(self.alphas, dim=-1)

    def determine_linear_beta_set(self):

        # Ensure that the beta values are within (0, 1) and that the total number of timesteps is positive and non-zero.
        assert self.beta_start > 0 and self.beta_start < 1
        assert self.beta_end > 0 and self.beta_end < 1
        assert self.total_timesteps > 0

        # Linearly interpolate beta values between beta_start and beta_end given total_diffusion_steps.
        return torch.linspace(self.beta_start, self.beta_end, self.total_timesteps)

    def forward_diffusion(self, x_0s, ts):

        """
        The forward diffusion formula is defined as:

            x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

        Where:

            - x_0 is the original non-noisy data sample.
            - epsilon is the noise sampled from a normal distribution.
            - alpha_bar_t is the cummalitive product of alpha values as alpha_0 * ... * alpha_t.

        """

        # Ensure the given diffusion timesteps are valid for the forward diffusion process.
        assert (ts >= 0).all()
        assert (ts < self.total_timesteps).all()

        device = x_0s.device
        alpha_bars_ts = self.alpha_bars[ts].to(device).view(-1, 1, 1, 1)

        # Compute sqrt(alpha_bar_t) * x_0.
        x_0_term = torch.mul(torch.sqrt(alpha_bars_ts), x_0s)

        # Sample random noise.
        epsilon = torch.randn_like(x_0s)

        # Compute sqrt(1 - alpha_bar_t) * epsilon.
        epsilon_term = torch.mul(torch.sqrt(torch.sub(1.0, alpha_bars_ts)), epsilon)

        # Add terms to obtain noisy sample x_t.
        x_ts = torch.add(x_0_term, epsilon_term)

        return x_ts, epsilon
    
    def reverse_diffusion(self, x_ts, ts, epsilon):

        """
        The reverse diffusion formula is defined as:

            x_t-1 = (1 / sqrt(alpha_t)) * (x_t - (1 - alpha_t) / sqrt(1 - alpha_bar_t) * epsilon) + sigma_t * z

        And sigma_t is defined as:

            sigma_t = sqrt((1 - alpha_bar_t-1) / (1 - alpha_bar_t) * beta_t)

        Where:

            - x_t is the noisy data sample at diffusion timestep t.
            - epsilon is the predicted noise.
            - z is the noise sampled from a normal distribution.
            - alpha_t/alpha_t-1 represents how much of the original signal is preserved at diffusion timestep t/t-1.
            - beta_t represents how much noise is added at diffusion timestep t.
            - alpha_bar_t is the cummalitive product of alpha values as alpha_0 * ... * alpha_t.

        """
        
        # Ensure the given diffusion timesteps are valid for the backwards diffusion process.
        assert (ts > 0).all()
        assert (ts < self.total_timesteps).all()
        
        device = x_ts.device
        alpha_bars_ts = self.alpha_bars[ts].to(device).view(-1, 1, 1, 1)
        alpha_bars_ts_prev = self.alpha_bars[ts - 1].to(device).view(-1, 1, 1, 1)
        alpha_ts = self.alphas[ts].to(device).view(-1, 1, 1, 1)
        betas_ts = self.betas[ts].to(device).view(-1, 1, 1, 1)

        # Sample random noise.
        z = torch.randn_like(x_ts)

        # Compute 1 / sqrt(alpha_t)
        scalar_term = torch.rsqrt(alpha_ts.to(device).view(-1, 1, 1, 1))

        # Compute (1 - alpha_t) / sqrt(1 - alpha_bar_t) * epsilon.
        noise_term = torch.mul(torch.div(torch.sub(1.0, alpha_ts), torch.sqrt(torch.sub(1.0, alpha_bars_ts))), epsilon)

        # Compute sigma_t.
        sigma_t = torch.sqrt(torch.mul(betas_ts, torch.div(torch.sub(1.0, alpha_bars_ts_prev), torch.sub(1.0, alpha_bars_ts))))

        # Compute sigma_t * z.
        sigma_term = torch.mul(z, sigma_t)

        x_t_prevs = torch.add(torch.mul(scalar_term, torch.sub(x_ts, noise_term)), sigma_term)

        return x_t_prevs
