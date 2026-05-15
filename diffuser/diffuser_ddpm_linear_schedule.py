import torch

class Diffuser_DDPM_linear_schedule:

    def __init__(self, total_timesteps=1000, beta_start=0.0001, beta_end=0.02):

        self.total_timesteps = total_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        self.betas = self.determine_linear_beta_set()
        self.alphas = torch.sub(1.0, self.betas)
        self.alpha_bars = torch.cumprod(self.alphas, dim=-1)

    def determine_linear_beta_set(self):

        assert self.beta_start > 0 and self.beta_start < 1
        assert self.beta_end > 0 and self.beta_end < 1
        assert self.total_timesteps > 0

        return torch.linspace(self.beta_start, self.beta_end, self.total_timesteps)

    def forward_diffusion(self, x_0s, ts):

        assert (ts >= 0).any()
        assert (ts < self.total_timesteps).any()

        device = x_0s.device

        alpha_bars_ts = self.alpha_bars[ts].to(device).view(-1, 1, 1, 1)

        mean_coef = torch.sqrt(alpha_bars_ts)
        var_coef = torch.sqrt(torch.sub(1.0, alpha_bars_ts))

        noise = torch.randn_like(x_0s)

        mean = torch.mul(mean_coef, x_0s)
        var = torch.mul(var_coef, noise)

        noised_x0s = torch.add(mean, var)

        return noised_x0s, noise
    
    def reverse_diffusion(self, x_ts, ts, predicted_noise):

        assert (ts > 0).any()
        assert (ts <= self.total_timesteps).any()
        
        device = x_ts.device

        alpha_bars_ts = self.alpha_bars[ts].to(device).view(-1, 1, 1, 1)
        alpha_bars_ts_prev = self.alpha_bars[ts - 1].to(device).view(-1, 1, 1, 1)
        betas_ts = self.betas[ts].to(device).view(-1, 1, 1, 1)

        noise = torch.randn_like(x_ts)

        variance = torch.mul(betas_ts, torch.div(torch.sub(1.0, alpha_bars_ts_prev), torch.sub(1.0, alpha_bars_ts)))

        sigma_z = torch.mul(noise, torch.sqrt(variance))
        
        noise_coef = torch.div(betas_ts, torch.sqrt(torch.sub(1.0, alpha_bars_ts)))
        
        reciprocal_sqrt_alpha_t = torch.rsqrt(self.alphas[ts].to(device).view(-1, 1, 1, 1))
        
        denoised_x_ts = torch.add(torch.mul(reciprocal_sqrt_alpha_t, torch.sub(x_ts, torch.mul(noise_coef, predicted_noise))), sigma_z)

        return denoised_x_ts
