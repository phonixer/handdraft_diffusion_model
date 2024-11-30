import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import polyline_encoder

class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, pred, target, weight=1.0):
        loss = self._loss(pred, target)
        WeightedLoss = (loss * weight).mean()
        return WeightedLoss

class WeightedL1(WeightedMSELoss):
    def _loss(self, pred, target):
        return torch.abs(pred - target)

class WeightedL2(WeightedMSELoss):
    def _loss(self, pred, target):
        return F.mse_loss(pred, target, reduction='none')

Losses = {
    'l1': WeightedL1,
    'l2': WeightedL2
}

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class MLPMap(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, device, t_dim, num_polylines, num_points_each_polylines, in_channels, num_layers, num_pre_layers, out_channels, mlp_hidden_dim, mlp_out_dim):
        super(MLPMap, self).__init__()

        self.t_dim = t_dim
        self.a_dim = action_dim
        self.device = device
        self.agent_polyline_encoder = polyline_encoder.PointNetPolylineEncoder(
            in_channels, hidden_dim, num_layers, num_pre_layers, out_channels)

        mlp_in_dim = num_polylines * out_channels
        print("mlp_in_dim:", mlp_in_dim)

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim)
        )

        input_dim = mlp_in_dim + action_dim + t_dim
        print("input_dim:", input_dim)
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
        )

        self.mid_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.Mish(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
        )
        self.final_layer = nn.Linear(2 * hidden_dim, action_dim)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, time, state, **kwargs):
        t_emb = self.time_mlp(time)
        polylines, polylines_mask = state['polylines'], state['polylines_mask']
        encoded_features = self.agent_polyline_encoder(polylines, polylines_mask)
        encoded_features = encoded_features.reshape(encoded_features.shape[0], -1)

        x = torch.cat([x, encoded_features, t_emb], dim=1)
        x1 = self.input_layer(x)
        x = self.mid_layer(x1)
        x = torch.cat([x, x1], dim=1)
        x = self.final_layer(x)
        return x

class Diffusion(nn.Module):
    def __init__(self, loss_type, beta_schedule, clip_denoised, predict_epsilon, obs_dim, act_dim, hidden_dim, device, T, t_dim, num_polylines, num_points_each_polylines, in_channels, num_layers, num_pre_layers, out_channels, mlp_hidden_dim, mlp_out_dim):
        super(Diffusion, self).__init__()

        self.state_dim = obs_dim
        self.action_dim = act_dim
        self.hidden_dim = hidden_dim
        self.device = torch.device(device)
        self.T = T
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon
        self.model = MLPMap(self.state_dim, self.action_dim, self.hidden_dim, self.device, t_dim, num_polylines, num_points_each_polylines, in_channels, num_layers, num_pre_layers, out_channels, mlp_hidden_dim, mlp_out_dim).to(device)

        if beta_schedule == 'linear':
            betas = torch.linspace(0.0001, 0.02, self.T, dtype=torch.float32, device=self.device)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, 0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=self.device), alphas_cumprod[:-1]], dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        posterior_variance = (betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))

        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1.0 - alphas_cumprod_prev) * torch.sqrt(betas) / (1.0 - alphas_cumprod))

        self.loss_fn = Losses[loss_type]()

    def q_posterior(self, x_start, x, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x.shape) * x
        )
        posterior_variance = extract(self.posterior_variance, t, x.shape)
        posterior_log_variance = extract(self.posterior_log_variance_clipped, t, x.shape)
        return posterior_mean, posterior_variance, posterior_log_variance

    def predict_start_from_noise(self, x, t, pred_noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - extract(self.sqrt_recipm_alphas_cumprod, t, x.shape) * pred_noise
        )

    def p_mean_variance(self, x, t, state):
        pred_noise = self.model(x, t, state)
        x_recon = self.predict_start_from_noise(x, t, pred_noise)
        if self.clip_denoised:
            pred_noise = torch.clamp(pred_noise, -1.0, 1.0)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_recon, x, t)
        return model_mean, posterior_log_variance

    def p_sample(self, x, t, state):
        batchsize, *_, device = *x.shape, x.device
        model_mean, model_log_variance = self.p_mean_variance(x, t, state)
        noise = torch.randn_like(x)
        nonzero_mask = (1 - (t == 0).float()).reshape(batchsize, *((1,) * (len(x.shape) - 1)))
        return model_mean + torch.exp(0.5 * model_log_variance) * noise * nonzero_mask

    def p_sample_loop(self, state, shape, *args, **kwargs):
        device = self.device
        batch_size = state['polylines'].shape[0]
        x = torch.randn(shape, device=device, requires_grad=False)
        self.diffusion_steps = []

        for i in reversed(range(0, self.T)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t, state)
            self.diffusion_steps.append(x.clone())

        return x

    def sample(self, state, *args, **kwargs):
        batch_size = state['polylines'].shape[0]
        shape = [batch_size, self.action_dim]
        action = self.p_sample_loop(state, shape, *args, **kwargs)
        return action, self.diffusion_steps

    def q_sample(self, x_start, t, noise):
        if noise is None:
            noise = torch.randn_like(x_start)
        sample = (extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                  extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)
        return sample

    def p_losses(self, x_start, state, t, weights=1.0):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.model(x_noisy, t, state)
        assert noise.shape == x_recon.shape
        polylines_mask = state['polylines_mask']
        print("x_recon Shape:", x_recon.shape)
        print("noise Shape:", noise.shape)
        print("polylines_mask:", polylines_mask.shape)
        batch_size = x_start.shape[0]
        weights = polylines_mask.reshape(batch_size, -1).float()
        print("weights Shape:", weights.shape)

        if self.predict_epsilon:
            loss = self.loss_fn(x_recon, noise, weights)
        else:
            loss = self.loss_fn(x_recon, x_start, weights)

        print("loss Shape:", loss.shape)
        return loss

    def loss(self, x, state, weights=1.0):
        batch_size = len(x)
        t = torch.randint(0, self.T, (batch_size,), device=self.device).long()
        return self.p_losses(x, state, t, weights)

    def forward(self, state, *args, **kwargs):
        return self.sample(state, *args, **kwargs)

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    act_dim = 160
    obs_dim = 11
    batch_size = 100
    num_polylines = 8
    num_points_each_polylines = 20
    in_channels = 2
    hidden_dim = 256
    T = 10
    loss_type = 'l2'
    beta_schedule = 'linear'
    clip_denoised = True
    predict_epsilon = True
    t_dim = 16
    num_layers = 3
    num_pre_layers = 1
    out_channels = 10
    mlp_hidden_dim = 256
    mlp_out_dim = 320

    polylines = torch.randn(batch_size, num_polylines, num_points_each_polylines, in_channels).to(device)
    polylines_mask = torch.randint(0, 2, (batch_size, num_polylines, num_points_each_polylines)).bool().to(device)

    x = torch.randn(batch_size, act_dim).to(device)
    state = {'polylines': polylines, 'polylines_mask': polylines_mask}

    model = Diffusion(
        loss_type=loss_type,
        beta_schedule=beta_schedule,
        clip_denoised=clip_denoised,
        predict_epsilon=predict_epsilon,
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=hidden_dim,
        device=device,
        T=T,
        t_dim=t_dim,
        num_polylines=num_polylines,
        num_points_each_polylines=num_points_each_polylines,
        in_channels=in_channels,
        num_layers=num_layers,
        num_pre_layers=num_pre_layers,
        out_channels=out_channels,
        mlp_hidden_dim=mlp_hidden_dim,
        mlp_out_dim=mlp_out_dim
    )
    result, diffusion_steps = model(state)

    loss = model.loss(x, state)
    print(f"action: {result}; loss: {loss.item()}")

    print("polylines:", polylines)

    import matplotlib.pyplot as plt
    import torch.optim as optim

    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    model.train()
    for i in range(10000):
        loss = model.loss(x, state)
        loss.backward()
        print(f"loss: {loss.item()}")
        optimizer.step()
        optimizer.zero_grad()

    state_test = {'polylines': polylines[0:1], 'polylines_mask': polylines_mask[0:1]}
    x_test = x[0:1]
    action, diffusion_steps = model.sample(state_test)

    loss = model.loss(x_test, state_test)
    print(f"action: {action}; loss: {loss.item()}")
    print(x_test)
    print(len(diffusion_steps))

    # Plot the polylines before applying the mask
    plt.figure(figsize=(10, 5))
    for i in range(num_polylines):
        plt.plot(polylines[0, i, :, 0].cpu().numpy(), polylines[0, i, :, 1].cpu().numpy(), label=f'Polyline {i+1}')
    plt.title('Polylines Before Applying Mask')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.savefig('before.png')

    # Apply the mask to remove the specific line
    masked_polylines = polylines.clone()
    masked_polylines[~polylines_mask] = float('nan')  # Set masked points to NaN for plotting

    # Plot the polylines after applying the mask
    plt.figure(figsize=(10, 5))
    for i in range(num_polylines):
        plt.plot(masked_polylines[0, i, :, 0].cpu().numpy(), masked_polylines[0, i, :, 1].cpu().numpy(), label=f'Polyline {i+1}')
    plt.title('Polylines After Applying Mask')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.savefig('after.png')

    num_steps = len(diffusion_steps)
    steps_to_plot = [int(i * num_steps / 10) for i in range(10)] + [num_steps - 1]
    x_test = x_test.cpu().detach().numpy().flatten()
    plt.figure(figsize=(15, 5))
    for step_idx in steps_to_plot:
        step = diffusion_steps[step_idx].cpu().detach().numpy().flatten()
        plt.scatter([step_idx] * len(step), step, label=f'Step {step_idx}')

    plt.scatter([steps_to_plot[-1]] * len(x_test), x_test, label='Ground Truth')
    plt.title('Diffusion Process')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig('diffusion.png')
    plt.show()